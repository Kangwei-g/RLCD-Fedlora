"""LoRA shape changes without reloading or reinitializing the frozen model."""

import math

import torch
from torch import nn


class ValidationFeedbackRank:
    """Bidirectional bounded rank heuristic using only server validation CE."""

    def __init__(self, initial_rank=10, min_rank=6, max_rank=10, step=1,
                 patience=5, decrease_patience=3, min_delta=1e-4, warmup_rounds=2):
        integers = [initial_rank, min_rank, max_rank, step, patience, decrease_patience, warmup_rounds]
        if any(isinstance(x, bool) or int(x) != x for x in integers):
            raise ValueError('Rank, step, patience and warmup must be integers')
        if (min(min_rank, step, patience, decrease_patience) < 1 or not min_rank <= initial_rank <= max_rank
                or min_rank >= max_rank or warmup_rounds < 0):
            raise ValueError('Require 0 < min_rank <= initial_rank <= max_rank, distinct bounds and positive step/patience')
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('min_delta must be finite and nonnegative')
        self.rank, self.min_rank, self.max_rank = initial_rank, min_rank, max_rank
        self.step, self.patience = step, patience
        self.decrease_patience = decrease_patience
        self.min_delta, self.warmup_rounds = min_delta, warmup_rounds
        self.best_loss, self.previous_loss, self.last_round = None, None, 0
        self.bad_rounds, self.worsening_rounds = 0, 0

    def rebase(self, server_validation_loss):
        """After resizing, compare training against this new model, not the old rank."""
        loss = float(server_validation_loss)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('Server validation loss must be finite and nonnegative')
        self.best_loss = self.previous_loss = loss
        self.bad_rounds = self.worsening_rounds = 0

    def observe(self, round_number, server_validation_loss):
        if round_number != self.last_round + 1:
            raise ValueError('Rank feedback must arrive once per round in order')
        loss = float(server_validation_loss)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('Server validation loss must be finite and nonnegative')
        self.last_round = round_number
        trained_rank = self.rank
        if round_number <= self.warmup_rounds:
            self.best_loss, self.bad_rounds, reason = loss, 0, 'factor_warmup'
            self.worsening_rounds = 0
        elif self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss, self.bad_rounds, reason = loss, 0, 'validation_improved'
            self.worsening_rounds = 0
        else:
            self.bad_rounds += 1
            self.worsening_rounds = (self.worsening_rounds + 1
                                    if self.previous_loss is not None and loss > self.previous_loss + self.min_delta else 0)
            reason = 'validation_plateau'
        observed_bad_rounds = self.bad_rounds
        observed_worsening_rounds = self.worsening_rounds
        observed_best_loss = self.best_loss
        self.previous_loss = loss
        if self.worsening_rounds >= self.decrease_patience:
            self.rank = max(self.rank - self.step, self.min_rank)
            reason = 'shrink_after_worsening' if self.rank < trained_rank else 'rank_floor'
        elif self.bad_rounds >= self.patience:
            self.rank = min(self.rank + self.step, self.max_rank)
            reason = 'grow_after_plateau' if self.rank > trained_rank else 'rank_cap'
        if self.rank != trained_rank:
            self.best_loss = self.previous_loss = None
            self.bad_rounds = self.worsening_rounds = 0
        return dict(round=round_number, rank=trained_rank, proposed_next_rank=self.rank,
                    server_validation_loss=loss, best_loss_at_rank=observed_best_loss,
                    consecutive_nonimproving_rounds=observed_bad_rounds,
                    consecutive_worsening_rounds=observed_worsening_rounds, reason=reason)


def lora_layout(model):
    return {name: {'rank': int(module.r), 'alpha': float(module.alpha)}
            for name, module in model.named_modules()
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b')}


def common_rank(model):
    ranks = {value['rank'] for value in lora_layout(model).values()}
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError('Expected one common LoRA rank for this controlled experiment')
    return ranks.pop()


def match_checkpoint_layout(model, state, layout=None):
    """Allocate adapter shapes before loading values; never alter backbone/head."""
    layout = layout or {}
    for name, module in model.named_modules():
        if not hasattr(module, 'lora_a') or not hasattr(module, 'lora_b'):
            continue
        a_name, b_name = name + '.lora_a', name + '.lora_b'
        if a_name not in state or b_name not in state:
            raise ValueError('Missing adapter pair: ' + name)
        a, b = state[a_name], state[b_name]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.shape[1] < 1:
            raise ValueError('Invalid adapter shapes: ' + name)
        if a.shape[0] != module.lora_a.shape[0] or b.shape[1] != module.lora_b.shape[1]:
            raise ValueError('Checkpoint backbone dimensions do not match: ' + name)
        rank = int(a.shape[1])
        metadata = layout.get(name, {'rank': rank, 'alpha': module.alpha})
        if int(metadata['rank']) != rank or float(metadata['alpha']) <= 0:
            raise ValueError('Invalid adapter metadata: ' + name)
        for suffix, value in [('lora_a', a), ('lora_b', b)]:
            current = getattr(module, suffix)
            if current.shape != value.shape:
                setattr(module, suffix, nn.Parameter(current.new_empty(value.shape),
                                                     requires_grad=current.requires_grad))
        module.r = rank
        module.alpha = float(metadata['alpha'])


@torch.no_grad()
def resize_common_rank(model, new_rank):
    """Expand product-preservingly, or truncate its SVD; alpha/r stays correct."""
    modules = [(name, module) for name, module in model.named_modules()
               if hasattr(module, 'lora_a') and hasattr(module, 'lora_b')]
    if not modules or isinstance(new_rank, bool) or int(new_rank) != new_rank or new_rank < 1:
        raise ValueError('A positive integer rank and a LoRA model are required')
    new_rank = int(new_rank)
    for name, module in modules:
        if new_rank > min(module.lora_a.shape[0], module.lora_b.shape[1]):
            raise ValueError('Rank exceeds adapter dimensions: ' + name)
    changes = []
    for name, module in modules:
        old_rank = module.r
        if old_rank == new_rank:
            continue
        a, b = module.lora_a, module.lora_b
        a_new = a.new_empty((a.shape[0], new_rank))
        b_new = b.new_zeros((new_rank, b.shape[1]))
        relative_error = 0.0
        if new_rank > old_rank:
            # Nonzero A / zero B makes new coordinates trainable after expansion.
            nn.init.kaiming_uniform_(a_new, a=math.sqrt(5))
            a_new[:, :old_rank].copy_(a)
            b_new[:old_rank].copy_(b * (new_rank / old_rank))
        else:
            dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
            qa, ra = torch.linalg.qr(a.to(dtype), mode='reduced')
            qb, rb = torch.linalg.qr(b.T.to(dtype), mode='reduced')
            u, s, vh = torch.linalg.svd(ra @ rb.T, full_matrices=False)
            if s.square().sum().item() == 0:
                nn.init.kaiming_uniform_(a_new, a=math.sqrt(5))
            else:
                root = (s[:new_rank] * (new_rank / old_rank)).clamp(min=0).sqrt()
                a_new.copy_((qa @ u[:, :new_rank]) * root)
                b_new.copy_(root[:, None] * (vh[:new_rank] @ qb.T))
                relative_error = float((s[new_rank:].square().sum() / s.square().sum()).sqrt())
        module.lora_a = nn.Parameter(a_new, requires_grad=a.requires_grad)
        module.lora_b = nn.Parameter(b_new, requires_grad=b.requires_grad)
        module.r = new_rank
        changes.append(dict(module=name, old_rank=old_rank, new_rank=new_rank,
                            relative_product_error=relative_error))
    return changes
