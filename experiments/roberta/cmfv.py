"""Conditional marginal factor valuation for federated LoRA.

CMFV values a layer-factor coordinate before local training. Task fit is an
optimization-aware estimate of one-step server-validation improvement under
the current LoRA parameterization. Cross-client coherence is measured after
training in the effective LoRA weight space and maintained as a discounted
estimate for future rounds. A/B choices are sampled from their posterior
advantage probability without a prescribed factor ratio.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Optional, Sequence

import torch
from torch import nn


_LAYER_RE = re.compile(r"(?:encoder\.layer\.|transformer\.h\.?)(\d+)\.")
_FACTORS = ("A", "B")
_UINT64_MASK = (1 << 64) - 1


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


def _coordinate_uniform(seed: int, round_one_based: int, layer_id: int) -> float:
    """Return a stateless, reproducible U[0, 1) draw for one coordinate."""
    value = (
        (int(seed) & _UINT64_MASK)
        ^ ((int(round_one_based) + 0x9E3779B9) * 0x9E3779B97F4A7C15)
        ^ ((int(layer_id) + 0x85EBCA6B) * 0xBF58476D1CE4E5B9)
    ) & _UINT64_MASK
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return float((value >> 11) * (1.0 / (1 << 53)))


def _lora_modules(model: nn.Module):
    modules = []
    for name, module in model.named_modules():
        if not all(hasattr(module, field) for field in ("lora_a", "lora_b", "raw_linear")):
            continue
        layer_id = _layer_from_name(name)
        raw_weight = getattr(getattr(module, "raw_linear"), "weight", None)
        if layer_id is None or raw_weight is None:
            continue
        modules.append((name, int(layer_id), module))
    return modules


def _forward_validation_loss(model, batch, device, finetune=None):
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
    return nn.functional.cross_entropy(outputs.logits, labels), int(labels.shape[0])


def _directional_curvature(
    model: nn.Module,
    batch,
    device,
    finetune,
    parameter_gradients: Sequence[tuple[nn.Parameter, torch.Tensor]],
    base_loss: float,
    fd_eps: float,
    eps: float,
) -> dict[str, float]:
    gradient_energy = float(
        sum(
            torch.sum(gradient.detach().double().square()).item()
            for _, gradient in parameter_gradients
        )
    )
    if gradient_energy <= float(eps):
        return {
            "gradient_energy": 0.0,
            "curvature": 0.0,
            "plus_loss": float(base_loss),
            "minus_loss": float(base_loss),
            "max_restore_error": 0.0,
        }

    gradient_norm = math.sqrt(gradient_energy)
    originals = [
        parameter.detach().clone()
        for parameter, _ in parameter_gradients
    ]
    directions = [
        gradient.detach().to(device=parameter.device, dtype=parameter.dtype)
        / gradient_norm
        for parameter, gradient in parameter_gradients
    ]
    plus_loss = float(base_loss)
    minus_loss = float(base_loss)
    try:
        with torch.no_grad():
            for (parameter, _), original, direction in zip(
                parameter_gradients, originals, directions
            ):
                parameter.copy_(original + float(fd_eps) * direction)
            plus_loss = float(
                _forward_validation_loss(
                    model, batch, device, finetune=finetune
                )[0].detach().item()
            )

            for (parameter, _), original, direction in zip(
                parameter_gradients, originals, directions
            ):
                parameter.copy_(original - float(fd_eps) * direction)
            minus_loss = float(
                _forward_validation_loss(
                    model, batch, device, finetune=finetune
                )[0].detach().item()
            )
    finally:
        with torch.no_grad():
            for (parameter, _), original in zip(
                parameter_gradients, originals
            ):
                parameter.copy_(original)

    max_restore_error = max(
        float(
            torch.max(
                torch.abs(parameter.detach().float() - original.float())
            ).item()
        )
        for (parameter, _), original in zip(parameter_gradients, originals)
    )
    curvature = (
        plus_loss - 2.0 * float(base_loss) + minus_loss
    ) / (float(fd_eps) ** 2)
    return {
        "gradient_energy": gradient_energy,
        "curvature": float(curvature),
        "plus_loss": plus_loss,
        "minus_loss": minus_loss,
        "max_restore_error": max_restore_error,
    }


def compute_server_task_fit(
    model: nn.Module,
    validation_loader,
    device,
    finetune=None,
    learning_rate: float = 5e-4,
    curvature_fd_eps: float = 1e-3,
    eps: float = 1e-12,
) -> dict:
    """Compute optimization-aware marginal task fit for every LoRA factor."""
    if float(learning_rate) <= 0.0:
        raise ValueError("CMFV learning_rate must be positive")
    if float(curvature_fd_eps) <= 0.0:
        raise ValueError("CMFV curvature_fd_eps must be positive")
    modules = _lora_modules(model)
    if not modules:
        raise RuntimeError("CMFV found no LoRA modules with raw_linear weights")

    named_parameters = dict(model.named_parameters())
    raw_weight_names = [f"{name}.raw_linear.weight" for name, _, _ in modules]
    factor_names = [
        f"{name}.lora_{factor.lower()}"
        for name, _, _ in modules
        for factor in _FACTORS
    ]
    target_names = set(raw_weight_names + factor_names)
    missing = [name for name in target_names if name not in named_parameters]
    if missing:
        raise RuntimeError(f"CMFV scoring parameters missing: {missing}")

    requires_grad_before = {
        name: bool(parameter.requires_grad)
        for name, parameter in named_parameters.items()
    }
    was_training = bool(model.training)
    original_device = next(model.parameters()).device
    total_examples = max(1, len(validation_loader.dataset))
    validation_loss_sum = 0.0
    curvature_batch = None
    max_restore_error = 0.0

    try:
        model.to(device)
        model.eval()
        model.zero_grad(set_to_none=True)
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(name in target_names)

        for batch in validation_loader:
            if curvature_batch is None:
                curvature_batch = tuple(
                    value.detach().cpu().clone() for value in batch
                )
            loss, batch_size = _forward_validation_loss(
                model, batch, device, finetune=finetune
            )
            validation_loss_sum += float(loss.detach().item()) * batch_size
            (loss * (batch_size / total_examples)).backward()

        if curvature_batch is None:
            raise RuntimeError("CMFV validation loader produced no batches")
        with torch.no_grad():
            curvature_base_loss = float(
                _forward_validation_loss(
                    model,
                    curvature_batch,
                    device,
                    finetune=finetune,
                )[0].detach().item()
            )

        per_layer = {
            layer_id: {
                "gradient_energy": 0.0,
                "module_count": 0,
                "factor_parameter_count_A": 0,
                "factor_parameter_count_B": 0,
            }
            for layer_id in sorted({layer_id for _, layer_id, _ in modules})
        }
        layer_factor_gradients = {
            layer_id: {factor: [] for factor in _FACTORS}
            for layer_id in per_layer
        }
        for module_name, layer_id, module in modules:
            raw_gradient = named_parameters[
                f"{module_name}.raw_linear.weight"
            ].grad
            if raw_gradient is None:
                raise RuntimeError(
                    f"Missing CMFV validation gradient for {module_name}.raw_linear.weight"
                )
            row = per_layer[layer_id]
            row["gradient_energy"] += float(
                torch.sum(raw_gradient.detach().double().square()).item()
            )
            row["module_count"] += 1
            for factor in _FACTORS:
                parameter = named_parameters[
                    f"{module_name}.lora_{factor.lower()}"
                ]
                if parameter.grad is None:
                    raise RuntimeError(
                        f"Missing CMFV factor gradient for {module_name}, factor={factor}"
                    )
                layer_factor_gradients[layer_id][factor].append(
                    (parameter, parameter.grad.detach())
                )
                row[f"factor_parameter_count_{factor}"] += int(
                    parameter.numel()
                )

        marginal_gains = []
        for layer_id, row in per_layer.items():
            for factor in _FACTORS:
                diagnostics = _directional_curvature(
                    model=model,
                    batch=curvature_batch,
                    device=device,
                    finetune=finetune,
                    parameter_gradients=layer_factor_gradients[layer_id][factor],
                    base_loss=curvature_base_loss,
                    fd_eps=curvature_fd_eps,
                    eps=eps,
                )
                max_restore_error = max(
                    max_restore_error,
                    float(diagnostics["max_restore_error"]),
                )
                stability = max(
                    0.0,
                    1.0
                    - 0.5
                    * float(learning_rate)
                    * float(diagnostics["curvature"]),
                )
                marginal_gain = (
                    float(diagnostics["gradient_energy"]) * stability
                )
                row[f"factor_gradient_energy_{factor}"] = float(
                    diagnostics["gradient_energy"]
                )
                row[f"directional_curvature_{factor}"] = float(
                    diagnostics["curvature"]
                )
                row[f"curvature_plus_loss_{factor}"] = float(
                    diagnostics["plus_loss"]
                )
                row[f"curvature_minus_loss_{factor}"] = float(
                    diagnostics["minus_loss"]
                )
                row[f"stability_multiplier_{factor}"] = float(stability)
                row[f"marginal_gain_{factor}"] = float(marginal_gain)
                marginal_gains.append(float(marginal_gain))

        normalization = max(marginal_gains, default=0.0)
        for layer_id, row in per_layer.items():
            if normalization <= float(eps):
                row["Q_A"] = 0.0
                row["Q_B"] = 0.0
            else:
                row["Q_A"] = max(
                    0.0,
                    min(1.0, float(row["marginal_gain_A"]) / normalization),
                )
                row["Q_B"] = max(
                    0.0,
                    min(1.0, float(row["marginal_gain_B"]) / normalization),
                )
            row["layer_id"] = int(layer_id)

        return {
            "validation_loss": float(validation_loss_sum / total_examples),
            "curvature_batch_loss": curvature_base_loss,
            "curvature_batch_examples": int(curvature_batch[2].shape[0]),
            "curvature_fd_eps": float(curvature_fd_eps),
            "learning_rate": float(learning_rate),
            "marginal_gain_normalization": float(normalization),
            "max_restore_error": float(max_restore_error),
            "num_examples": int(total_examples),
            "num_modules": int(len(modules)),
            "per_layer": per_layer,
        }
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(requires_grad_before[name])
        model.train(was_training)
        model.to(original_device)


def compute_client_factor_coherence(
    client_payloads: Sequence[Mapping[str, torch.Tensor]],
    state_before: Mapping[str, torch.Tensor],
    factor_map: Mapping[int, str],
    aggregation_weights: Sequence[float],
    eps: float = 1e-12,
) -> dict[int, dict[str, float]]:
    """Measure cancellation of selected client updates in effective AB space."""
    if len(client_payloads) != len(aggregation_weights):
        raise ValueError(
            f"CMFV received {len(client_payloads)} payloads but "
            f"{len(aggregation_weights)} aggregation weights"
        )
    weight_total = float(sum(float(value) for value in aggregation_weights))
    if weight_total <= 0.0:
        raise ValueError("CMFV aggregation weights must have a positive sum")
    weights = [float(value) / weight_total for value in aggregation_weights]

    output = {}
    for layer_id, selected_factor in factor_map.items():
        layer_id = int(layer_id)
        selected_factor = str(selected_factor).upper()
        client_effective_updates = []
        parameter_count = 0

        for payload in client_payloads:
            module_updates = []
            for name, client_value in payload.items():
                if (
                    _layer_from_name(name) != layer_id
                    or _factor_from_name(name) != selected_factor
                ):
                    continue
                before = state_before.get(name)
                if before is None:
                    continue
                base_name = name.rsplit(".", 1)[0]
                delta = (
                    client_value.detach().to(dtype=torch.float64, device="cpu")
                    - before.detach().to(dtype=torch.float64, device="cpu")
                )
                if selected_factor == "A":
                    companion = state_before.get(f"{base_name}.lora_b")
                    if companion is None:
                        continue
                    effective_delta = delta @ companion.to(
                        dtype=torch.float64, device="cpu"
                    )
                else:
                    companion = state_before.get(f"{base_name}.lora_a")
                    if companion is None:
                        continue
                    effective_delta = companion.to(
                        dtype=torch.float64, device="cpu"
                    ) @ delta
                module_updates.append(effective_delta)
            if not module_updates:
                raise RuntimeError(
                    f"CMFV found no client update for layer={layer_id}, "
                    f"factor={selected_factor}"
                )
            parameter_count = max(parameter_count, len(module_updates))
            client_effective_updates.append(module_updates)

        if any(len(updates) != parameter_count for updates in client_effective_updates):
            raise RuntimeError(
                f"CMFV inconsistent projection count for layer={layer_id}, "
                f"factor={selected_factor}"
            )

        numerator = 0.0
        denominator = 0.0
        aggregate_norm_sq = 0.0
        client_norm_sq = []
        for module_idx in range(parameter_count):
            aggregate_update = sum(
                weight * client_effective_updates[client_idx][module_idx]
                for client_idx, weight in enumerate(weights)
            )
            module_aggregate_norm_sq = float(
                torch.sum(aggregate_update.square()).item()
            )
            numerator += module_aggregate_norm_sq
            aggregate_norm_sq += module_aggregate_norm_sq
            for client_idx, weight in enumerate(weights):
                value = float(
                    torch.sum(
                        client_effective_updates[client_idx][module_idx].square()
                    ).item()
                )
                denominator += weight * value
                if len(client_norm_sq) <= client_idx:
                    client_norm_sq.append(0.0)
                client_norm_sq[client_idx] += value

        coherence = numerator / (denominator + float(eps)) if denominator > 0.0 else 0.0
        coherence = max(0.0, min(1.0, float(coherence)))
        output[layer_id] = {
            "factor": selected_factor,
            "coherence": coherence,
            "aggregate_norm": math.sqrt(max(0.0, aggregate_norm_sq)),
            "weighted_client_energy": float(denominator),
            "mean_client_norm": float(
                sum(weight * math.sqrt(max(0.0, value))
                    for weight, value in zip(weights, client_norm_sq))
            ),
            "num_clients": int(len(client_payloads)),
            "parameter_count": int(parameter_count),
        }
    return output


class CMFVSelector:
    """Natural joint layer-factor selector based on posterior value advantage."""

    def __init__(
        self,
        total_layers: int,
        coherence_prior: float = 0.5,
        ema_decay: float = 0.9,
        count_discount: float = 0.97,
        confidence_z: float = 1.0,
        min_task_fit: float = 1e-10,
        exploration_beta: float = 0.05,
        posterior_temperature: float = 0.5,
        seed: int = 0,
    ):
        if not 0.0 <= float(coherence_prior) <= 1.0:
            raise ValueError("coherence_prior must be in [0, 1]")
        if not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 < float(count_discount) <= 1.0:
            raise ValueError("count_discount must be in (0, 1]")
        if float(confidence_z) < 0.0:
            raise ValueError("confidence_z must be non-negative")
        if float(min_task_fit) < 0.0:
            raise ValueError("min_task_fit must be non-negative")
        if float(exploration_beta) < 0.0:
            raise ValueError("exploration_beta must be non-negative")
        if float(posterior_temperature) <= 0.0:
            raise ValueError("posterior_temperature must be positive")

        self.total_layers = int(total_layers)
        self.coherence_prior = float(coherence_prior)
        self.ema_decay = float(ema_decay)
        self.count_discount = float(count_discount)
        self.confidence_z = float(confidence_z)
        self.min_task_fit = float(min_task_fit)
        self.exploration_beta = float(exploration_beta)
        self.posterior_temperature = float(posterior_temperature)
        self.seed = int(seed)
        self.coherence = {
            layer_id: {factor: self.coherence_prior for factor in _FACTORS}
            for layer_id in range(self.total_layers)
        }
        self.variance = {
            layer_id: {factor: 0.25 for factor in _FACTORS}
            for layer_id in range(self.total_layers)
        }
        self.effective_counts = {
            layer_id: {factor: 0.0 for factor in _FACTORS}
            for layer_id in range(self.total_layers)
        }
        self.counts = {
            layer_id: {factor: 0 for factor in _FACTORS}
            for layer_id in range(self.total_layers)
        }
        self.last_observed_round = {
            layer_id: {factor: None for factor in _FACTORS}
            for layer_id in range(self.total_layers)
        }

    def _discounted_stats(
        self,
        layer_id: int,
        factor: str,
        round_one_based: int,
    ) -> tuple[float, float, float]:
        last_round = self.last_observed_round[layer_id][factor]
        age = (
            max(0, int(round_one_based) - int(last_round))
            if last_round is not None else 0
        )
        discount = self.count_discount ** age
        effective_count = self.effective_counts[layer_id][factor] * discount
        if self.counts[layer_id][factor] == 0:
            mean = self.coherence_prior
        else:
            mean = (
                self.coherence_prior
                + discount
                * (self.coherence[layer_id][factor] - self.coherence_prior)
            )
        variance = min(
            0.25,
            max(0.0, self.variance[layer_id][factor] + 0.25 * (1.0 - discount)),
        )
        return float(mean), float(variance), float(effective_count)

    def _factor_metrics(
        self,
        layer_id: int,
        factor: str,
        task_fit: float,
        round_one_based: int,
    ) -> dict[str, float]:
        task_fit = max(0.0, min(1.0, float(task_fit)))
        mean, variance, effective_count = self._discounted_stats(
            layer_id,
            factor,
            round_one_based,
        )
        epistemic = 0.5 / math.sqrt(effective_count + 1.0)
        volatility = math.sqrt(max(0.0, variance) / (effective_count + 1.0))
        uncertainty = min(0.5, max(epistemic, volatility))
        lower_c = max(0.0, mean - self.confidence_z * uncertainty)
        upper_c = min(1.0, mean + self.confidence_z * uncertainty)
        factor_value = task_fit * mean
        value_uncertainty = task_fit * uncertainty
        exploration_bonus = self.exploration_beta * value_uncertainty
        return {
            "task_fit": task_fit,
            "coherence": mean,
            "coherence_variance": variance,
            "effective_count": effective_count,
            "count": int(self.counts[layer_id][factor]),
            "uncertainty": uncertainty,
            "value_uncertainty": value_uncertainty,
            "factor_value": factor_value,
            "exploration_bonus": exploration_bonus,
            "decision_score": factor_value + exploration_bonus,
            "lower_bound": task_fit * lower_c,
            "upper_bound": task_fit * upper_c,
        }

    def _a_probability(
        self,
        metrics: Mapping[str, Mapping[str, float]],
    ) -> tuple[float, float]:
        """Return P(value_A > value_B) under the current value posterior."""
        task_fit_a = float(metrics["A"]["task_fit"])
        task_fit_b = float(metrics["B"]["task_fit"])
        if task_fit_a <= self.min_task_fit < task_fit_b:
            return 0.0, float("-inf")
        if task_fit_b <= self.min_task_fit < task_fit_a:
            return 1.0, float("inf")

        value_a = float(metrics["A"]["factor_value"])
        value_b = float(metrics["B"]["factor_value"])
        variance = (
            float(metrics["A"]["value_uncertainty"]) ** 2
            + float(metrics["B"]["value_uncertainty"]) ** 2
        )
        if variance <= 1e-24:
            if value_a > value_b:
                return 1.0, float("inf")
            if value_b > value_a:
                return 0.0, float("-inf")
            return 0.5, 0.0

        denominator = self.posterior_temperature * math.sqrt(variance)
        posterior_z = (value_a - value_b) / max(denominator, 1e-12)
        probability_a = 0.5 * (
            1.0 + math.erf(posterior_z / math.sqrt(2.0))
        )
        return (
            max(0.0, min(1.0, float(probability_a))),
            float(posterior_z),
        )

    def select_coordinates(
        self,
        task_fit_by_layer: Mapping[int, Mapping[str, float]],
        budget: int,
        round_one_based: int,
    ) -> tuple[list[int], dict[int, str], dict[int, dict]]:
        budget = max(1, min(int(budget), self.total_layers))
        metadata = {}
        for layer_id in range(self.total_layers):
            task_row = task_fit_by_layer.get(layer_id, {})
            factor_metrics = {
                "A": self._factor_metrics(
                    layer_id, "A", task_row.get("Q_A", 0.0), round_one_based
                ),
                "B": self._factor_metrics(
                    layer_id, "B", task_row.get("Q_B", 0.0), round_one_based
                ),
            }
            probability_a, posterior_z = self._a_probability(factor_metrics)
            expected_factor_value = (
                probability_a * float(factor_metrics["A"]["factor_value"])
                + (1.0 - probability_a)
                * float(factor_metrics["B"]["factor_value"])
            )
            expected_decision_score = (
                probability_a * float(factor_metrics["A"]["decision_score"])
                + (1.0 - probability_a)
                * float(factor_metrics["B"]["decision_score"])
            )
            metadata[layer_id] = {
                "layer_id": layer_id,
                "selected_factor": "B",
                "decision_reason": "not_selected",
                "exploration": False,
                "probability_A": probability_a,
                "factor_random_draw": None,
                "posterior_z": posterior_z,
                "greedy_factor": (
                    "A"
                    if factor_metrics["A"]["factor_value"]
                    > factor_metrics["B"]["factor_value"]
                    else "B"
                ),
                "factor_value_A": factor_metrics["A"]["factor_value"],
                "factor_value_B": factor_metrics["B"]["factor_value"],
                "layer_value": expected_factor_value,
                "layer_decision_score": expected_decision_score,
                "A": factor_metrics["A"],
                "B": factor_metrics["B"],
            }

        selected_layers = sorted(
            range(self.total_layers),
            key=lambda layer_id: (
                -float(metadata[layer_id]["layer_decision_score"]),
                -float(metadata[layer_id]["layer_value"]),
                int(layer_id),
            ),
        )[:budget]
        factor_map = {}
        for layer_id in selected_layers:
            draw = _coordinate_uniform(self.seed, round_one_based, layer_id)
            probability_a = float(metadata[layer_id]["probability_A"])
            selected_factor = "A" if draw < probability_a else "B"
            factor_map[int(layer_id)] = selected_factor
            metadata[layer_id]["selected_factor"] = selected_factor
            metadata[layer_id]["factor_random_draw"] = draw
            metadata[layer_id]["decision_reason"] = (
                "posterior_sample_a"
                if selected_factor == "A"
                else "posterior_sample_b"
            )
            metadata[layer_id]["exploration"] = (
                selected_factor != metadata[layer_id]["greedy_factor"]
            )
        selected_set = set(selected_layers)
        for layer_id in range(self.total_layers):
            metadata[layer_id]["selected"] = layer_id in selected_set
        return selected_layers, factor_map, metadata

    def update_coherence(
        self,
        factor_map: Mapping[int, str],
        observations: Mapping[int, Mapping[str, float]],
        round_one_based: int,
    ) -> dict[int, dict[str, float]]:
        updates = {}
        for layer_id, factor in factor_map.items():
            layer_id = int(layer_id)
            factor = str(factor).upper()
            if layer_id not in observations:
                raise RuntimeError(
                    f"Missing CMFV coherence observation for layer={layer_id}"
                )
            observed = max(
                0.0,
                min(1.0, float(observations[layer_id]["coherence"])),
            )
            mean_before, variance_before, effective_before = self._discounted_stats(
                layer_id,
                factor,
                round_one_based,
            )
            if self.counts[layer_id][factor] == 0:
                mean_after = observed
                variance_after = 0.0
            else:
                mean_after = (
                    self.ema_decay * mean_before
                    + (1.0 - self.ema_decay) * observed
                )
                residual = observed - mean_before
                variance_after = (
                    self.ema_decay * variance_before
                    + (1.0 - self.ema_decay) * residual * residual
                )
            effective_after = effective_before + 1.0
            self.coherence[layer_id][factor] = float(mean_after)
            self.variance[layer_id][factor] = float(
                max(0.0, min(0.25, variance_after))
            )
            self.effective_counts[layer_id][factor] = float(effective_after)
            self.counts[layer_id][factor] += 1
            self.last_observed_round[layer_id][factor] = int(round_one_based)
            updates[layer_id] = {
                "factor": factor,
                "observed_coherence": observed,
                "coherence_before": mean_before,
                "coherence_after": float(mean_after),
                "variance_before": variance_before,
                "variance_after": float(self.variance[layer_id][factor]),
                "effective_count_before": effective_before,
                "effective_count_after": effective_after,
                "count_after": int(self.counts[layer_id][factor]),
            }
        return updates

    def state_dict(self) -> dict:
        return {
            "total_layers": self.total_layers,
            "coherence_prior": self.coherence_prior,
            "ema_decay": self.ema_decay,
            "count_discount": self.count_discount,
            "confidence_z": self.confidence_z,
            "min_task_fit": self.min_task_fit,
            "exploration_beta": self.exploration_beta,
            "posterior_temperature": self.posterior_temperature,
            "seed": self.seed,
            "coherence": self.coherence,
            "variance": self.variance,
            "effective_counts": self.effective_counts,
            "counts": self.counts,
            "last_observed_round": self.last_observed_round,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["total_layers"]) != self.total_layers:
            raise ValueError(
                f"CMFV layer mismatch: checkpoint={state['total_layers']} "
                f"runtime={self.total_layers}"
            )
        self.coherence_prior = float(state["coherence_prior"])
        self.ema_decay = float(state["ema_decay"])
        self.count_discount = float(state["count_discount"])
        self.confidence_z = float(state["confidence_z"])
        self.min_task_fit = float(state["min_task_fit"])
        self.exploration_beta = float(
            state.get("exploration_beta", self.exploration_beta)
        )
        self.posterior_temperature = float(
            state.get("posterior_temperature", self.posterior_temperature)
        )
        self.seed = int(state.get("seed", self.seed))

        def restore_factor_state(name, caster):
            raw = state[name]
            return {
                int(layer_id): {
                    factor: caster(value) if value is not None else None
                    for factor, value in payload.items()
                }
                for layer_id, payload in raw.items()
            }

        self.coherence = restore_factor_state("coherence", float)
        self.variance = restore_factor_state("variance", float)
        self.effective_counts = restore_factor_state("effective_counts", float)
        self.counts = restore_factor_state("counts", int)
        self.last_observed_round = restore_factor_state(
            "last_observed_round", int
        )
