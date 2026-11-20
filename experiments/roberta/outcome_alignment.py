"""Outcome-aligned LoRA factor selection for ARC-FedLoRA.

This module deliberately does not use AS-LoRA finite-difference scoring. It
scores the factor that was actually aggregated by comparing its update with
the server-validation loss gradient, then combines that alignment with the
round-level validation reward.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Optional

import torch
from torch import nn


_LAYER_RE = re.compile(r"(?:encoder\.layer\.|transformer\.h\.?)(\d+)\.")


def _layer_from_name(name: str) -> Optional[int]:
    match = _LAYER_RE.search(str(name))
    return int(match.group(1)) if match else None


def _factor_from_name(name: str) -> Optional[str]:
    suffix = str(name).rsplit(".", 1)[-1].lower()
    if suffix == "lora_a":
        return "A"
    if suffix == "lora_b":
        return "B"
    return None


def selected_factor_parameter_names(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    factor_map: Mapping[int, str],
) -> list[str]:
    normalized_map = {
        int(layer_id): str(factor).upper()
        for layer_id, factor in factor_map.items()
    }
    selected = []
    for name, _ in named_parameters:
        layer_id = _layer_from_name(name)
        factor = _factor_from_name(name)
        if layer_id in normalized_map and factor == normalized_map[layer_id]:
            selected.append(name)
    return selected


def summarize_factor_alignment(
    state_before: Mapping[str, torch.Tensor],
    state_after: Mapping[str, torch.Tensor],
    validation_gradients: Mapping[str, torch.Tensor],
    factor_map: Mapping[int, str],
    eps: float = 1e-12,
) -> dict[int, dict[str, float]]:
    """Return per-layer cosine alignment between -validation gradient and update."""
    diagnostics = {}
    for layer_id, factor in factor_map.items():
        layer_id = int(layer_id)
        factor = str(factor).upper()
        dot = 0.0
        grad_sq = 0.0
        delta_sq = 0.0
        numel = 0
        parameter_count = 0

        for name, after in state_after.items():
            if _layer_from_name(name) != layer_id or _factor_from_name(name) != factor:
                continue
            before = state_before.get(name)
            gradient = validation_gradients.get(name)
            if before is None or gradient is None:
                continue
            delta = (
                after.detach().to(dtype=torch.float64, device="cpu")
                - before.detach().to(dtype=torch.float64, device="cpu")
            )
            gradient = gradient.detach().to(dtype=torch.float64, device="cpu")
            if delta.shape != gradient.shape:
                raise ValueError(
                    f"Gradient/update shape mismatch for {name}: "
                    f"gradient={tuple(gradient.shape)}, delta={tuple(delta.shape)}"
                )
            dot += float(torch.sum(gradient * delta).item())
            grad_sq += float(torch.sum(gradient * gradient).item())
            delta_sq += float(torch.sum(delta * delta).item())
            numel += int(delta.numel())
            parameter_count += 1

        grad_norm = math.sqrt(max(0.0, grad_sq))
        delta_norm = math.sqrt(max(0.0, delta_sq))
        denominator = grad_norm * delta_norm
        alignment = -dot / (denominator + float(eps)) if denominator > 0.0 else 0.0
        alignment = max(-1.0, min(1.0, float(alignment)))
        diagnostics[layer_id] = {
            "factor": factor,
            "alignment": alignment,
            "gradient_update_dot": float(dot),
            "gradient_norm": float(grad_norm),
            "delta_norm": float(delta_norm),
            "numel": int(numel),
            "parameter_count": int(parameter_count),
        }
    return diagnostics


def compute_server_validation_alignment(
    model: nn.Module,
    validation_loader,
    state_before: Mapping[str, torch.Tensor],
    factor_map: Mapping[int, str],
    device,
    finetune=None,
    eps: float = 1e-12,
) -> dict:
    """Compute validation gradients and align them with the aggregated LoRA update."""
    named_parameters = dict(model.named_parameters())
    target_names = selected_factor_parameter_names(
        named_parameters.items(),
        factor_map,
    )
    if not target_names:
        raise RuntimeError(
            f"No LoRA parameters matched outcome-alignment factor_map={dict(factor_map)}"
        )

    expected_parameter_count = 2 * len(factor_map)
    if len(target_names) != expected_parameter_count:
        raise RuntimeError(
            "Outcome alignment expected key/value parameters for each selected factor: "
            f"expected={expected_parameter_count}, found={len(target_names)}, "
            f"factor_map={dict(factor_map)}"
        )

    requires_grad_before = {
        name: bool(parameter.requires_grad)
        for name, parameter in named_parameters.items()
    }
    was_training = bool(model.training)
    first_parameter = next(model.parameters())
    original_device = first_parameter.device
    target_set = set(target_names)
    total_examples = max(1, len(validation_loader.dataset))
    validation_loss_sum = 0.0

    try:
        model.to(device)
        model.eval()
        model.zero_grad(set_to_none=True)
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(name in target_set)

        for batch in validation_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            if finetune is not None and finetune != "full":
                outputs = model.backbone_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            loss = nn.functional.cross_entropy(outputs.logits, labels)
            batch_size = int(labels.shape[0])
            validation_loss_sum += float(loss.detach().item()) * batch_size
            (loss * (batch_size / total_examples)).backward()

        gradients = {}
        state_after = {}
        for name in target_names:
            parameter = named_parameters[name]
            if parameter.grad is None:
                raise RuntimeError(
                    f"Missing server-validation gradient for selected parameter {name}"
                )
            gradients[name] = parameter.grad.detach().cpu().clone()
            state_after[name] = parameter.detach().cpu().clone()

        per_layer = summarize_factor_alignment(
            state_before=state_before,
            state_after=state_after,
            validation_gradients=gradients,
            factor_map=factor_map,
            eps=eps,
        )
        return {
            "validation_loss": float(validation_loss_sum / total_examples),
            "num_examples": int(total_examples),
            "num_parameters": int(len(target_names)),
            "per_layer": per_layer,
        }
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(requires_grad_before[name])
        model.train(was_training)
        model.to(original_device)


class OutcomeAlignmentFactorSelector:
    """Per-layer A/B utility estimator with observation-driven warm-up and UCB."""

    def __init__(
        self,
        total_layers: int,
        min_observations: int = 2,
        ema_decay: float = 0.9,
        alignment_weight: float = 0.7,
        ucb_coefficient: float = 0.25,
        reward_scale_decay: float = 0.9,
        credit_clip: float = 1.0,
    ):
        if int(min_observations) < 1:
            raise ValueError("min_observations must be at least 1")
        if not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 <= float(alignment_weight) <= 1.0:
            raise ValueError("alignment_weight must be in [0, 1]")
        if float(ucb_coefficient) < 0.0:
            raise ValueError("ucb_coefficient must be non-negative")
        if not 0.0 <= float(reward_scale_decay) < 1.0:
            raise ValueError("reward_scale_decay must be in [0, 1)")
        if float(credit_clip) <= 0.0:
            raise ValueError("credit_clip must be positive")

        self.total_layers = int(total_layers)
        self.min_observations = int(min_observations)
        self.ema_decay = float(ema_decay)
        self.alignment_weight = float(alignment_weight)
        self.ucb_coefficient = float(ucb_coefficient)
        self.reward_scale_decay = float(reward_scale_decay)
        self.credit_clip = float(credit_clip)
        self.utilities = {
            layer_id: {"A": 0.0, "B": 0.0}
            for layer_id in range(self.total_layers)
        }
        self.counts = {
            layer_id: {"A": 0, "B": 0}
            for layer_id in range(self.total_layers)
        }
        self.reward_abs_ema = 0.0
        self.rounds_updated = 0

    def _ucb_value(self, layer_id: int, factor: str) -> float:
        count = self.counts[layer_id][factor]
        total = self.counts[layer_id]["A"] + self.counts[layer_id]["B"]
        if count <= 0:
            return float("inf")
        bonus = self.ucb_coefficient * math.sqrt(
            math.log(float(total) + 1.0) / float(count)
        )
        return float(self.utilities[layer_id][factor] + bonus)

    def select_factor_map(self, active_layers: Iterable[int]):
        factor_map = {}
        metadata = {}
        for layer_id in sorted({int(value) for value in active_layers}):
            count_a = self.counts[layer_id]["A"]
            count_b = self.counts[layer_id]["B"]
            if count_a < self.min_observations or count_b < self.min_observations:
                if count_b <= count_a and count_b < self.min_observations:
                    factor = "B"
                else:
                    factor = "A"
                reason = "minimum_observation"
                ucb_a = self._ucb_value(layer_id, "A")
                ucb_b = self._ucb_value(layer_id, "B")
            else:
                ucb_a = self._ucb_value(layer_id, "A")
                ucb_b = self._ucb_value(layer_id, "B")
                if ucb_a > ucb_b:
                    factor = "A"
                elif ucb_b > ucb_a:
                    factor = "B"
                else:
                    factor = "B" if count_b <= count_a else "A"
                reason = "ucb"

            factor_map[layer_id] = factor
            metadata[layer_id] = {
                "reason": reason,
                "utility_a": float(self.utilities[layer_id]["A"]),
                "utility_b": float(self.utilities[layer_id]["B"]),
                "count_a": int(count_a),
                "count_b": int(count_b),
                "ucb_a": float(ucb_a),
                "ucb_b": float(ucb_b),
            }
        return factor_map, metadata

    def update(
        self,
        factor_map: Mapping[int, str],
        alignment_diagnostics: Mapping[int, Mapping[str, float]],
        validation_reward: float,
    ) -> dict:
        reward = float(validation_reward)
        reward_abs = abs(reward)
        if self.rounds_updated == 0:
            self.reward_abs_ema = reward_abs
        else:
            self.reward_abs_ema = (
                self.reward_scale_decay * self.reward_abs_ema
                + (1.0 - self.reward_scale_decay) * reward_abs
            )
        reward_scale = max(self.reward_abs_ema, 1e-6)
        normalized_reward = max(-1.0, min(1.0, reward / reward_scale))

        alignments = {
            int(layer_id): float(
                alignment_diagnostics.get(int(layer_id), {}).get("alignment", 0.0)
            )
            for layer_id in factor_map
        }
        nonzero_magnitudes = [
            abs(value) for value in alignments.values() if abs(value) > 0.0
        ]
        mean_alignment_magnitude = (
            sum(nonzero_magnitudes) / len(nonzero_magnitudes)
            if nonzero_magnitudes else 1.0
        )

        updates = {}
        for layer_id, selected_factor in factor_map.items():
            layer_id = int(layer_id)
            selected_factor = str(selected_factor).upper()
            alignment = alignments[layer_id]
            attribution_weight = min(
                2.0,
                abs(alignment) / max(mean_alignment_magnitude, 1e-12),
            )
            global_component = normalized_reward * attribution_weight
            credit = (
                self.alignment_weight * alignment
                + (1.0 - self.alignment_weight) * global_component
            )
            credit = max(-self.credit_clip, min(self.credit_clip, credit))

            utility_before = self.utilities[layer_id][selected_factor]
            utility_after = (
                self.ema_decay * utility_before
                + (1.0 - self.ema_decay) * credit
            )
            self.utilities[layer_id][selected_factor] = float(utility_after)
            self.counts[layer_id][selected_factor] += 1
            updates[layer_id] = {
                "factor": selected_factor,
                "alignment": float(alignment),
                "attribution_weight": float(attribution_weight),
                "normalized_reward": float(normalized_reward),
                "global_component": float(global_component),
                "credit": float(credit),
                "utility_before": float(utility_before),
                "utility_after": float(utility_after),
                "count_after": int(self.counts[layer_id][selected_factor]),
            }

        self.rounds_updated += 1
        return {
            "reward": reward,
            "reward_abs_ema": float(self.reward_abs_ema),
            "normalized_reward": float(normalized_reward),
            "mean_alignment_magnitude": float(mean_alignment_magnitude),
            "layers": updates,
        }

    def state_dict(self) -> dict:
        return {
            "total_layers": self.total_layers,
            "min_observations": self.min_observations,
            "ema_decay": self.ema_decay,
            "alignment_weight": self.alignment_weight,
            "ucb_coefficient": self.ucb_coefficient,
            "reward_scale_decay": self.reward_scale_decay,
            "credit_clip": self.credit_clip,
            "utilities": self.utilities,
            "counts": self.counts,
            "reward_abs_ema": self.reward_abs_ema,
            "rounds_updated": self.rounds_updated,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["total_layers"]) != self.total_layers:
            raise ValueError(
                f"Selector layer mismatch: checkpoint={state['total_layers']} "
                f"runtime={self.total_layers}"
            )
        self.min_observations = int(state["min_observations"])
        self.ema_decay = float(state["ema_decay"])
        self.alignment_weight = float(state["alignment_weight"])
        self.ucb_coefficient = float(state["ucb_coefficient"])
        self.reward_scale_decay = float(state["reward_scale_decay"])
        self.credit_clip = float(state["credit_clip"])
        self.utilities = {
            int(layer_id): {
                factor: float(value)
                for factor, value in payload.items()
            }
            for layer_id, payload in state["utilities"].items()
        }
        self.counts = {
            int(layer_id): {
                factor: int(value)
                for factor, value in payload.items()
            }
            for layer_id, payload in state["counts"].items()
        }
        self.reward_abs_ema = float(state.get("reward_abs_ema", 0.0))
        self.rounds_updated = int(state.get("rounds_updated", 0))
