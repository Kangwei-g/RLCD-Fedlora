"""AS-LoRA component selection and finite-difference scoring for ARC-FedLoRA.

The selector follows the released AS-LoRA implementation where it differs from
the paper: EMA beta is the decay on the previous score and post-warmup logits
use the relative A/B score difference before temperature scaling.
"""

from __future__ import annotations

import math
import random
import re
import threading
from typing import Dict, Iterable, Mapping, Optional

import torch


OFFICIAL_EMA_BETA = 0.9
OFFICIAL_SCORE_EPS = 1e-8
OFFICIAL_FD_EPS = 1e-3
OFFICIAL_T0 = 1.0
OFFICIAL_TMIN = 0.1
OFFICIAL_GAMMA = 0.95

_LAYER_RE = re.compile(r"\.layer\.(\d+)\.")
_PROJECTION_CACHE = {}
_PROJECTION_LOCK = threading.Lock()


def _factor_from_name(name: str) -> Optional[str]:
    suffix = name.rsplit(".", 1)[-1]
    if suffix == "lora_a":
        return "A"
    if suffix == "lora_b":
        return "B"
    return None


def _layer_from_name(name: str) -> Optional[int]:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _projection_seed(base_seed: int, layer_id: int, factor: str, dim: int) -> int:
    factor_offset = 17 if factor == "A" else 53
    return int(base_seed) + int(layer_id) * 100_003 + factor_offset * 1_009 + int(dim)


def _get_gaussian_projection(
    layer_id: int,
    factor: str,
    dim: int,
    device: torch.device,
    base_seed: int,
) -> torch.Tensor:
    key = (int(base_seed), int(layer_id), str(factor), int(dim), str(device))
    with _PROJECTION_LOCK:
        cached = _PROJECTION_CACHE.get(key)
        if cached is not None:
            return cached

        generator = torch.Generator(device="cpu")
        generator.manual_seed(_projection_seed(base_seed, layer_id, factor, dim))
        projection = torch.randn(
            (dim, dim), generator=generator, dtype=torch.float32, device="cpu"
        ) / math.sqrt(float(dim))
        projection = projection.to(device=device)
        _PROJECTION_CACHE[key] = projection
        return projection


def _project_gradient(
    gradient: torch.Tensor,
    layer_id: int,
    factor: str,
    projection_seed: int,
) -> torch.Tensor:
    gradient = gradient.detach().float()
    if gradient.ndim != 2:
        raise ValueError(
            f"AS-LoRA expects 2-D LoRA gradients, got shape={tuple(gradient.shape)}"
        )

    # This repository stores A as [d_in, r] and B as [r, d_out], transposed
    # relative to PEFT. Project the feature/channel dimension accordingly.
    if factor == "A":
        dim = int(gradient.shape[0])
        projection = _get_gaussian_projection(
            layer_id, factor, dim, gradient.device, projection_seed
        )
        return projection @ gradient

    dim = int(gradient.shape[1])
    projection = _get_gaussian_projection(
        layer_id, factor, dim, gradient.device, projection_seed
    )
    return gradient @ projection


def _forward_loss(model, batch, criterion) -> torch.Tensor:
    input_ids, attention_mask, labels = batch
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return criterion(outputs.logits, labels)


def compute_arc_aslora_fd_scores(
    model,
    batch,
    criterion,
    active_layers: Iterable[int],
    eta: float,
    fd_eps: float = OFFICIAL_FD_EPS,
    norm_eps: float = OFFICIAL_SCORE_EPS,
    projection_seed: int = 0,
):
    """Compute A and B scores without changing model, gradients, or optimizer state."""
    active_layers = sorted({int(layer_id) for layer_id in active_layers})
    scores = {
        layer_id: {"A": 0.0, "B": 0.0}
        for layer_id in active_layers
    }
    named_params = dict(model.named_parameters())
    target_names = []
    for name in named_params:
        layer_id = _layer_from_name(name)
        factor = _factor_from_name(name)
        if layer_id in scores and factor is not None:
            target_names.append(name)

    expected_targets = len(active_layers) * 4
    if len(target_names) != expected_targets:
        raise RuntimeError(
            "AS-LoRA scoring expected key/value A/B parameters for every active layer: "
            f"expected={expected_targets}, found={len(target_names)}, layers={active_layers}"
        )

    requires_grad_before = {
        name: bool(param.requires_grad) for name, param in named_params.items()
    }
    grads_before = {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in named_params.items()
    }
    was_training = bool(model.training)
    max_restore_error = 0.0

    try:
        model.eval()
        target_set = set(target_names)
        for name, param in named_params.items():
            param.requires_grad_(name in target_set)
            param.grad = None

        base_loss_tensor = _forward_loss(model, batch, criterion)
        base_loss_tensor.backward()
        base_loss = float(base_loss_tensor.detach().float().item())

        projected_grads = {}
        for name in target_names:
            param = named_params[name]
            if param.grad is None:
                raise RuntimeError(f"AS-LoRA score gradient is missing for {name}")
            layer_id = _layer_from_name(name)
            factor = _factor_from_name(name)
            projected_grads[name] = _project_gradient(
                param.grad, layer_id, factor, projection_seed
            )

        for name in target_names:
            param = named_params[name]
            layer_id = _layer_from_name(name)
            factor = _factor_from_name(name)
            projected_grad = projected_grads[name]
            grad_norm = projected_grad.norm()
            grad_sq = float(torch.sum(projected_grad * projected_grad).item())
            if grad_sq <= 1e-24:
                continue

            direction = projected_grad / (grad_norm + float(norm_eps))
            direction = direction.to(device=param.device, dtype=param.dtype)
            original = param.detach().clone()
            try:
                with torch.no_grad():
                    param.copy_(original + float(fd_eps) * direction)
                    plus_loss = float(_forward_loss(model, batch, criterion).detach().float().item())
                    param.copy_(original - float(fd_eps) * direction)
                    minus_loss = float(_forward_loss(model, batch, criterion).detach().float().item())
            finally:
                with torch.no_grad():
                    param.copy_(original)

            restore_error = float(
                torch.max(torch.abs(param.detach().float() - original.float())).item()
            )
            max_restore_error = max(max_restore_error, restore_error)
            curvature = (
                plus_loss - 2.0 * base_loss + minus_loss
            ) / (float(fd_eps) ** 2)
            score = grad_sq - 0.5 * float(eta) * curvature * grad_sq
            scores[layer_id][factor] += float(score)

        if max_restore_error != 0.0:
            raise RuntimeError(
                f"AS-LoRA finite-difference perturbation was not exactly restored: {max_restore_error}"
            )
    finally:
        for name, param in named_params.items():
            param.requires_grad_(requires_grad_before[name])
            previous_grad = grads_before[name]
            param.grad = None if previous_grad is None else previous_grad
        model.train(was_training)

    return scores, {
        "base_loss": base_loss,
        "max_restore_error": max_restore_error,
        "num_scored_parameters": len(target_names),
    }


class ASLoRAComponentSelector:
    """Server-side AS-LoRA EMA and temperature-softmax component selector."""

    def __init__(
        self,
        total_layers: int,
        warmup_rounds: int,
        ema_beta: float = OFFICIAL_EMA_BETA,
        temperature_0: float = OFFICIAL_T0,
        temperature_min: float = OFFICIAL_TMIN,
        temperature_gamma: float = OFFICIAL_GAMMA,
        score_eps: float = OFFICIAL_SCORE_EPS,
        seed: int = 0,
    ):
        self.total_layers = int(total_layers)
        self.warmup_rounds = int(warmup_rounds)
        self.ema_beta = float(ema_beta)
        self.temperature_0 = float(temperature_0)
        self.temperature_min = float(temperature_min)
        self.temperature_gamma = float(temperature_gamma)
        self.score_eps = float(score_eps)
        self.ema_scores = {
            layer_id: {"A": None, "B": None}
            for layer_id in range(self.total_layers)
        }
        self.random_state = random.Random(int(seed))
        self.last_temperature = self.temperature_0

    def temperature(self, round_one_based: int) -> float:
        if int(round_one_based) <= self.warmup_rounds:
            value = self.temperature_0
        else:
            exponent = int(round_one_based) - self.warmup_rounds
            value = max(
                self.temperature_min,
                self.temperature_0 * (self.temperature_gamma ** exponent),
            )
        self.last_temperature = float(value)
        return float(value)

    def probability_a(self, layer_id: int, temperature: float) -> float:
        layer_scores = self.ema_scores[int(layer_id)]
        score_a = layer_scores["A"]
        score_b = layer_scores["B"]
        if score_a is None and score_b is None:
            return 0.5
        if score_a is None:
            return 0.0
        if score_b is None:
            return 1.0

        relative_difference = (score_a - score_b) / (
            abs(score_a) + abs(score_b) + self.score_eps
        )
        logit = relative_difference / max(float(temperature), self.score_eps)
        logit = max(min(logit, 30.0), -30.0)
        return float(1.0 / (1.0 + math.exp(-logit)))

    def select_factor_map(self, active_layers: Iterable[int], round_one_based: int):
        active_layers = sorted({int(layer_id) for layer_id in active_layers})
        temperature = self.temperature(round_one_based)
        factor_map = {}
        metadata = {}

        if int(round_one_based) <= self.warmup_rounds:
            factor = "B" if int(round_one_based) % 2 == 1 else "A"
            for layer_id in active_layers:
                factor_map[layer_id] = factor
                metadata[layer_id] = {
                    "reason": "warmup_alternation",
                    "probability_a": 1.0 if factor == "A" else 0.0,
                    "random_draw": None,
                    "temperature": temperature,
                }
            return factor_map, metadata

        for layer_id in active_layers:
            probability_a = self.probability_a(layer_id, temperature)
            draw = self.random_state.random()
            factor = "A" if draw < probability_a else "B"
            factor_map[layer_id] = factor
            metadata[layer_id] = {
                "reason": "official_relative_softmax",
                "probability_a": probability_a,
                "random_draw": draw,
                "temperature": temperature,
            }
        return factor_map, metadata

    def update_from_clients(
        self,
        client_scores: Mapping[int, Mapping[int, Mapping[str, float]]],
        active_layers: Iterable[int],
    ):
        active_layers = sorted({int(layer_id) for layer_id in active_layers})
        if not client_scores:
            raise ValueError("AS-LoRA requires scores from at least one participating client")

        aggregated = {}
        for layer_id in active_layers:
            aggregated[layer_id] = {}
            for factor in ("A", "B"):
                values = []
                for client_id, layer_scores in client_scores.items():
                    if layer_id not in layer_scores or factor not in layer_scores[layer_id]:
                        raise ValueError(
                            f"Client {client_id} is missing AS-LoRA score for layer={layer_id}, factor={factor}"
                        )
                    value = float(layer_scores[layer_id][factor])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Client {client_id} produced non-finite AS-LoRA score {value} "
                            f"for layer={layer_id}, factor={factor}"
                        )
                    values.append(value)
                mean_score = float(sum(values) / len(values))
                aggregated[layer_id][factor] = mean_score

                previous = self.ema_scores[layer_id][factor]
                if previous is None:
                    updated = mean_score
                else:
                    updated = (
                        self.ema_beta * float(previous)
                        + (1.0 - self.ema_beta) * mean_score
                    )
                self.ema_scores[layer_id][factor] = float(updated)
        return aggregated

    def state_dict(self) -> dict:
        return {
            "total_layers": self.total_layers,
            "warmup_rounds": self.warmup_rounds,
            "ema_beta": self.ema_beta,
            "temperature_0": self.temperature_0,
            "temperature_min": self.temperature_min,
            "temperature_gamma": self.temperature_gamma,
            "score_eps": self.score_eps,
            "ema_scores": self.ema_scores,
            "random_state": self.random_state.getstate(),
            "last_temperature": self.last_temperature,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["total_layers"]) != self.total_layers:
            raise ValueError(
                f"AS-LoRA checkpoint layers={state['total_layers']} != current {self.total_layers}"
            )
        self.warmup_rounds = int(state["warmup_rounds"])
        self.ema_beta = float(state["ema_beta"])
        self.temperature_0 = float(state["temperature_0"])
        self.temperature_min = float(state["temperature_min"])
        self.temperature_gamma = float(state["temperature_gamma"])
        self.score_eps = float(state["score_eps"])
        self.ema_scores = {
            int(layer_id): {
                factor: None if value is None else float(value)
                for factor, value in factor_scores.items()
            }
            for layer_id, factor_scores in state["ema_scores"].items()
        }
        self.random_state.setstate(state["random_state"])
        self.last_temperature = float(state.get("last_temperature", self.temperature_0))
