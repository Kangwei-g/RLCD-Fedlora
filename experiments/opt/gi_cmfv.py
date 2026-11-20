"""Gauge-invariant counterfactual marginal factor valuation for federated LoRA.

GI-CMFV treats B as the within-subspace exploitation coordinate and values A
only by the additional effective-weight direction that B cannot express. Both
coordinates are scored on every participating client's validation data, even
though only one coordinate is trained. The server combines current
gauge-invariant task fit with empirical client evidence and samples the factor
without a prescribed A/B ratio or factor prior.
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


def _coordinate_uniform(seed: int, round_one_based: int, layer_id: int) -> float:
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


def _orthonormal_column_basis(
    matrix: torch.Tensor,
    rank_rtol: float,
) -> torch.Tensor:
    matrix = matrix.detach().to(dtype=torch.float64)
    if matrix.numel() == 0:
        return matrix.new_zeros((matrix.shape[0], 0))
    u, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0].item()) == 0.0:
        return matrix.new_zeros((matrix.shape[0], 0))
    threshold = float(rank_rtol) * float(singular_values[0].item())
    rank = int(torch.count_nonzero(singular_values > threshold).item())
    return u[:, :rank]


def _orthonormal_row_basis(
    matrix: torch.Tensor,
    rank_rtol: float,
) -> torch.Tensor:
    matrix = matrix.detach().to(dtype=torch.float64)
    if matrix.numel() == 0:
        return matrix.new_zeros((0, matrix.shape[1]))
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0].item()) == 0.0:
        return matrix.new_zeros((0, matrix.shape[1]))
    threshold = float(rank_rtol) * float(singular_values[0].item())
    rank = int(torch.count_nonzero(singular_values > threshold).item())
    return vh[:rank, :]


def gauge_invariant_coordinate_values(
    effective_gradient: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    rank_rtol: float = 1e-5,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return effective-space B exploitation and A revision values.

    For Delta W = A B, B updates span {A dB}. A updates span {dA B}.
    The B value is the gradient captured by col(A). The A value retains only
    the part of the A tangent outside col(A), because the overlapping part can
    already be represented by a B update:

        V_B = ||P_A G||_F^2
        V_A = ||(I - P_A) G P_B||_F^2

    P_A and P_B depend only on the represented subspaces, so the values are
    invariant to an invertible low-rank basis change A -> A R, B -> R^-1 B.
    """
    if float(rank_rtol) <= 0.0:
        raise ValueError("rank_rtol must be positive")
    gradient = effective_gradient.detach().to(dtype=torch.float64)
    a = lora_a.detach().to(device=gradient.device, dtype=torch.float64)
    b = lora_b.detach().to(device=gradient.device, dtype=torch.float64)
    if gradient.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("GI-CMFV expects 2-D gradient, A, and B tensors")
    if tuple(gradient.shape) != (int(a.shape[0]), int(b.shape[1])):
        raise ValueError(
            "GI-CMFV shape mismatch: "
            f"G={tuple(gradient.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
        )
    if int(a.shape[1]) != int(b.shape[0]):
        raise ValueError(
            f"GI-CMFV rank mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}"
        )

    gradient_energy = float(torch.sum(gradient.square()).item())
    if gradient_energy <= float(eps):
        return {
            "A": 0.0,
            "B": 0.0,
            "A_full": 0.0,
            "A_overlap": 0.0,
            "gradient_energy": 0.0,
            "rank_A": 0,
            "rank_B": 0,
        }

    basis_a = _orthonormal_column_basis(a, rank_rtol)
    basis_b = _orthonormal_row_basis(b, rank_rtol)

    if basis_a.shape[1] == 0:
        b_energy = 0.0
    else:
        b_coordinates = basis_a.transpose(0, 1) @ gradient
        b_energy = float(torch.sum(b_coordinates.square()).item())

    if basis_b.shape[0] == 0:
        a_full_energy = 0.0
        a_overlap_energy = 0.0
        a_revision_energy = 0.0
    else:
        a_tangent_coordinates = gradient @ basis_b.transpose(0, 1)
        a_full_energy = float(torch.sum(a_tangent_coordinates.square()).item())
        if basis_a.shape[1] == 0:
            a_overlap_energy = 0.0
            a_revision = a_tangent_coordinates
        else:
            overlap_coordinates = (
                basis_a.transpose(0, 1) @ a_tangent_coordinates
            )
            a_overlap_energy = float(
                torch.sum(overlap_coordinates.square()).item()
            )
            a_revision = (
                a_tangent_coordinates
                - basis_a @ overlap_coordinates
            )
        a_revision_energy = float(torch.sum(a_revision.square()).item())

    def fraction(value: float) -> float:
        return max(0.0, min(1.0, float(value) / gradient_energy))

    return {
        "A": fraction(a_revision_energy),
        "B": fraction(b_energy),
        "A_full": fraction(a_full_energy),
        "A_overlap": fraction(a_overlap_energy),
        "gradient_energy": gradient_energy,
        "rank_A": int(basis_a.shape[1]),
        "rank_B": int(basis_b.shape[0]),
    }


def _lora_modules(model: nn.Module, active_layers: set[int]):
    modules = []
    for name, module in model.named_modules():
        if not all(
            hasattr(module, field)
            for field in ("lora_a", "lora_b", "raw_linear")
        ):
            continue
        layer_id = _layer_from_name(name)
        raw_weight = getattr(getattr(module, "raw_linear"), "weight", None)
        if layer_id in active_layers and raw_weight is not None:
            modules.append((name, int(layer_id), module))
    return modules


def _forward_loss(model, batch, device, finetune=None) -> torch.Tensor:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)
    target = model
    if (
        finetune is not None
        and finetune != "full"
        and hasattr(model, "backbone_model")
    ):
        target = model.backbone_model
    outputs = target(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return nn.functional.cross_entropy(outputs.logits, labels)


def compute_gi_cmfv_scores(
    model: nn.Module,
    batch,
    active_layers: Iterable[int],
    device,
    finetune=None,
    rank_rtol: float = 1e-5,
    eps: float = 1e-12,
) -> tuple[dict[int, dict[str, float]], dict]:
    """Score both factors on one validation batch without changing the model."""
    active_layers = {int(layer_id) for layer_id in active_layers}
    if not active_layers:
        raise ValueError("GI-CMFV requires at least one active layer")
    modules = _lora_modules(model, active_layers)
    observed_layers = {layer_id for _, layer_id, _ in modules}
    missing_layers = sorted(active_layers - observed_layers)
    if missing_layers:
        raise RuntimeError(
            f"GI-CMFV found no LoRA modules for layers={missing_layers}"
        )

    named_parameters = dict(model.named_parameters())
    raw_names = [f"{name}.raw_linear.weight" for name, _, _ in modules]
    requires_grad_before = {
        name: bool(parameter.requires_grad)
        for name, parameter in named_parameters.items()
    }
    raw_grads_before = {
        name: (
            None
            if named_parameters[name].grad is None
            else named_parameters[name].grad.detach().clone()
        )
        for name in raw_names
    }
    was_training = bool(model.training)

    try:
        model.eval()
        raw_name_set = set(raw_names)
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(name in raw_name_set)
        for name in raw_names:
            named_parameters[name].grad = None

        loss = _forward_loss(
            model=model,
            batch=batch,
            device=device,
            finetune=finetune,
        )
        loss.backward()

        accumulators = {
            layer_id: {
                "A_energy": 0.0,
                "B_energy": 0.0,
                "A_full_energy": 0.0,
                "A_overlap_energy": 0.0,
                "gradient_energy": 0.0,
                "module_count": 0,
                "rank_A_sum": 0,
                "rank_B_sum": 0,
            }
            for layer_id in sorted(active_layers)
        }
        for module_name, layer_id, module in modules:
            raw_gradient = named_parameters[
                f"{module_name}.raw_linear.weight"
            ].grad
            if raw_gradient is None:
                raise RuntimeError(
                    f"GI-CMFV validation gradient missing for {module_name}"
                )
            diagnostics = gauge_invariant_coordinate_values(
                effective_gradient=raw_gradient.detach().transpose(0, 1),
                lora_a=module.lora_a,
                lora_b=module.lora_b,
                rank_rtol=rank_rtol,
                eps=eps,
            )
            weight = float(diagnostics["gradient_energy"])
            row = accumulators[layer_id]
            row["A_energy"] += float(diagnostics["A"]) * weight
            row["B_energy"] += float(diagnostics["B"]) * weight
            row["A_full_energy"] += float(diagnostics["A_full"]) * weight
            row["A_overlap_energy"] += (
                float(diagnostics["A_overlap"]) * weight
            )
            row["gradient_energy"] += weight
            row["module_count"] += 1
            row["rank_A_sum"] += int(diagnostics["rank_A"])
            row["rank_B_sum"] += int(diagnostics["rank_B"])

        scores = {}
        for layer_id, row in accumulators.items():
            denominator = float(row["gradient_energy"])
            if denominator <= float(eps):
                score_a = score_b = a_full = a_overlap = 0.0
            else:
                score_a = float(row["A_energy"]) / denominator
                score_b = float(row["B_energy"]) / denominator
                a_full = float(row["A_full_energy"]) / denominator
                a_overlap = float(row["A_overlap_energy"]) / denominator
            scores[layer_id] = {
                "A": max(0.0, min(1.0, score_a)),
                "B": max(0.0, min(1.0, score_b)),
                "A_full": max(0.0, min(1.0, a_full)),
                "A_overlap": max(0.0, min(1.0, a_overlap)),
                "gradient_energy": denominator,
                "module_count": int(row["module_count"]),
                "rank_A_mean": (
                    float(row["rank_A_sum"]) / max(1, row["module_count"])
                ),
                "rank_B_mean": (
                    float(row["rank_B_sum"]) / max(1, row["module_count"])
                ),
            }
        return scores, {
            "validation_loss": float(loss.detach().item()),
            "active_layers": sorted(active_layers),
            "num_modules": int(len(modules)),
            "rank_rtol": float(rank_rtol),
            "evidence": "gauge_invariant_effective_weight_tangent",
        }
    finally:
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(requires_grad_before[name])
        for name, previous_grad in raw_grads_before.items():
            named_parameters[name].grad = previous_grad
        model.train(was_training)


def compute_server_gi_cmfv_scores(
    model: nn.Module,
    validation_loader,
    active_layers: Iterable[int],
    device,
    finetune=None,
    rank_rtol: float = 1e-5,
) -> tuple[dict[int, dict[str, float]], dict]:
    """Compute current GI-CMFV task fit on deterministic server validation."""
    iterator = iter(validation_loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("GI-CMFV server validation loader is empty") from exc

    original_device = next(model.parameters()).device
    try:
        model.to(device)
        return compute_gi_cmfv_scores(
            model=model,
            batch=batch,
            active_layers=active_layers,
            device=device,
            finetune=finetune,
            rank_rtol=rank_rtol,
        )
    finally:
        model.to(original_device)


class GICMFVSelector:
    """Disagreement-aware evidence selector with no factor prior or ratio."""

    def __init__(
        self,
        total_layers: int,
        ema_decay: float = 0.9,
        count_discount: float = 0.98,
        warmup_rounds: int = 0,
        server_weight: float = 0.2,
        disagreement_penalty: float = 0.2,
        temperature_0: float = 1.0,
        temperature_min: float = 0.1,
        temperature_gamma: float = 0.95,
        seed: int = 0,
        eps: float = 1e-12,
    ):
        if not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 < float(count_discount) <= 1.0:
            raise ValueError("count_discount must be in (0, 1]")
        if int(warmup_rounds) < 0:
            raise ValueError("warmup_rounds must be non-negative")
        if not 0.0 <= float(server_weight) <= 1.0:
            raise ValueError("server_weight must be in [0, 1]")
        if float(disagreement_penalty) < 0.0:
            raise ValueError("disagreement_penalty must be non-negative")
        if float(temperature_0) <= 0.0 or float(temperature_min) <= 0.0:
            raise ValueError("GI-CMFV temperatures must be positive")
        if float(temperature_min) > float(temperature_0):
            raise ValueError("temperature_min cannot exceed temperature_0")
        if not 0.0 < float(temperature_gamma) <= 1.0:
            raise ValueError("temperature_gamma must be in (0, 1]")

        self.total_layers = int(total_layers)
        self.ema_decay = float(ema_decay)
        self.count_discount = float(count_discount)
        self.warmup_rounds = int(warmup_rounds)
        self.server_weight = float(server_weight)
        self.disagreement_penalty = float(disagreement_penalty)
        self.temperature_0 = float(temperature_0)
        self.temperature_min = float(temperature_min)
        self.temperature_gamma = float(temperature_gamma)
        self.seed = int(seed)
        self.eps = float(eps)
        self.means = {
            layer_id: {factor: None for factor in _FACTORS}
            for layer_id in range(self.total_layers)
        }
        self.variances = {
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

    def temperature(self, round_one_based: int) -> float:
        adaptive_round = max(
            1, int(round_one_based) - self.warmup_rounds
        )
        exponent = adaptive_round - 1
        return max(
            self.temperature_min,
            self.temperature_0 * (self.temperature_gamma ** exponent),
        )

    def _normalize_pair(self, score_a: float, score_b: float) -> dict[str, float]:
        score_a = max(0.0, float(score_a))
        score_b = max(0.0, float(score_b))
        total = score_a + score_b
        if total <= self.eps:
            return {"A": 0.5, "B": 0.5}
        return {"A": score_a / total, "B": score_b / total}

    def _discounted_evidence(
        self,
        layer_id: int,
        factor: str,
        round_one_based: int,
    ) -> dict[str, float]:
        count = int(self.counts[layer_id][factor])
        last_round = self.last_observed_round[layer_id][factor]
        age = (
            max(0, int(round_one_based) - int(last_round) - 1)
            if last_round is not None
            else 0
        )
        discount = self.count_discount ** age
        effective_count = float(count) * discount
        if count == 0:
            evidence_mean = None
            evidence_variance = 0.0
            uncertainty = 0.0
        else:
            evidence_mean = float(self.means[layer_id][factor])
            evidence_variance = min(
                0.25,
                float(self.variances[layer_id][factor])
                + 0.25 * (1.0 - discount),
            )
            uncertainty = math.sqrt(
                max(
                    0.0,
                    evidence_variance / max(effective_count, self.eps),
                )
            )
        return {
            "evidence_available": count > 0,
            "evidence_mean": (
                None
                if evidence_mean is None
                else max(0.0, min(1.0, float(evidence_mean)))
            ),
            "client_variance": max(
                0.0, float(evidence_variance)
            ),
            "client_disagreement": math.sqrt(
                max(0.0, float(evidence_variance))
            ),
            "uncertainty": float(uncertainty),
            "effective_count": float(effective_count),
            "count": count,
            "age": int(age),
        }

    def select_factor_map(
        self,
        active_layers: Iterable[int],
        server_scores: Mapping[int, Mapping[str, float]],
        round_one_based: int,
    ) -> tuple[dict[int, str], dict[int, dict]]:
        factor_map = {}
        metadata = {}
        for layer_id in sorted({int(value) for value in active_layers}):
            raw_server = server_scores.get(layer_id)
            if raw_server is None:
                raise RuntimeError(
                    f"GI-CMFV server score missing for layer={layer_id}"
                )
            server_share = self._normalize_pair(
                raw_server.get("A", 0.0),
                raw_server.get("B", 0.0),
            )
            factor_rows = {}
            for factor in _FACTORS:
                evidence = self._discounted_evidence(
                    layer_id, factor, round_one_based
                )
                client_reliability = min(
                    1.0, float(evidence["effective_count"])
                )
                client_weight = (
                    (1.0 - self.server_weight) * client_reliability
                    if evidence["evidence_available"]
                    else 0.0
                )
                server_weight = 1.0 - client_weight
                client_mean = (
                    float(evidence["evidence_mean"])
                    if evidence["evidence_available"]
                    else 0.0
                )
                expected_value = (
                    server_weight * server_share[factor]
                    + client_weight * client_mean
                )
                # A changes the represented low-rank subspace, so ambiguous
                # cross-client evidence is a revision risk. B exploits the
                # incumbent subspace and remains the conservative fallback.
                disagreement_penalty = (
                    self.disagreement_penalty
                    * evidence["client_disagreement"]
                    if factor == "A"
                    else 0.0
                )
                risk_adjusted_value = max(
                    0.0,
                    expected_value - disagreement_penalty,
                )
                factor_rows[factor] = {
                    **evidence,
                    "server_value": float(raw_server.get(factor, 0.0)),
                    "server_share": float(server_share[factor]),
                    "server_weight_applied": float(server_weight),
                    "client_weight_applied": float(client_weight),
                    "expected_value": float(expected_value),
                    "disagreement_penalty_applied": float(
                        disagreement_penalty
                    ),
                    "risk_adjusted_value": float(risk_adjusted_value),
                }

            difference = (
                factor_rows["A"]["risk_adjusted_value"]
                - factor_rows["B"]["risk_adjusted_value"]
            )
            relative_difference = difference / (
                abs(factor_rows["A"]["risk_adjusted_value"])
                + abs(factor_rows["B"]["risk_adjusted_value"])
                + self.eps
            )
            temperature = self.temperature(round_one_based)
            value_a = factor_rows["A"]["risk_adjusted_value"]
            value_b = factor_rows["B"]["risk_adjusted_value"]
            warmup_active = int(round_one_based) <= self.warmup_rounds
            if warmup_active:
                selected_factor = (
                    "B" if int(round_one_based) % 2 == 1 else "A"
                )
                probability_a = 1.0 if selected_factor == "A" else 0.0
                selection_logit = None
                decision_reason = "warmup_alternation"
                random_draw = None
            elif value_a <= self.eps < value_b:
                selection_logit = -30.0
                probability_a = 0.0
                decision_reason = "zero_A_value_exclusion"
                random_draw = _coordinate_uniform(
                    self.seed, round_one_based, layer_id
                )
            elif value_b <= self.eps < value_a:
                selection_logit = 30.0
                probability_a = 1.0
                decision_reason = "zero_B_value_exclusion"
                random_draw = _coordinate_uniform(
                    self.seed, round_one_based, layer_id
                )
            else:
                selection_logit = relative_difference / temperature
                selection_logit = max(-30.0, min(30.0, selection_logit))
                probability_a = 1.0 / (1.0 + math.exp(-selection_logit))
                decision_reason = "prior_free_relative_softmax"
                random_draw = _coordinate_uniform(
                    self.seed, round_one_based, layer_id
                )
            if not warmup_active:
                selected_factor = (
                    "A" if random_draw < probability_a else "B"
                )
            greedy_factor = "A" if difference > 0.0 else "B"
            factor_map[layer_id] = selected_factor
            metadata[layer_id] = {
                "layer_id": int(layer_id),
                "selected_factor": selected_factor,
                "greedy_factor": greedy_factor,
                "probability_A": float(probability_a),
                "relative_difference": float(relative_difference),
                "selection_logit": (
                    None
                    if selection_logit is None
                    else float(selection_logit)
                ),
                "temperature": float(temperature),
                "random_draw": (
                    None
                    if random_draw is None
                    else float(random_draw)
                ),
                "warmup_active": bool(warmup_active),
                "schedule_override": bool(warmup_active),
                "exploration": (
                    False
                    if warmup_active
                    else selected_factor != greedy_factor
                ),
                "decision_reason": decision_reason,
                "A": factor_rows["A"],
                "B": factor_rows["B"],
                "server_A_full": float(raw_server.get("A_full", 0.0)),
                "server_A_overlap": float(
                    raw_server.get("A_overlap", 0.0)
                ),
            }
        return factor_map, metadata

    def update_from_clients(
        self,
        client_scores: Mapping[int, Mapping[int, Mapping[str, float]]],
        active_layers: Iterable[int],
        round_one_based: int,
    ) -> dict[int, dict]:
        if not client_scores:
            raise ValueError(
                "GI-CMFV requires evidence from participating clients"
            )
        output = {}
        for layer_id in sorted({int(value) for value in active_layers}):
            normalized_by_client = {}
            for client_id, scores_by_layer in client_scores.items():
                if layer_id not in scores_by_layer:
                    raise ValueError(
                        f"Client {client_id} lacks GI-CMFV layer={layer_id}"
                    )
                row = scores_by_layer[layer_id]
                normalized_by_client[int(client_id)] = self._normalize_pair(
                    row.get("A", 0.0), row.get("B", 0.0)
                )

            output[layer_id] = {}
            for factor in _FACTORS:
                values = [
                    row[factor] for row in normalized_by_client.values()
                ]
                round_mean = float(sum(values) / len(values))
                round_variance = float(
                    sum((value - round_mean) ** 2 for value in values)
                    / len(values)
                )
                count_before = int(self.counts[layer_id][factor])
                mean_before = self.means[layer_id][factor]
                variance_before = float(self.variances[layer_id][factor])
                if count_before == 0:
                    mean_after = round_mean
                    variance_after = round_variance
                else:
                    mean_after = (
                        self.ema_decay * float(mean_before)
                        + (1.0 - self.ema_decay) * round_mean
                    )
                    temporal_residual = round_mean - float(mean_before)
                    variance_after = (
                        self.ema_decay * variance_before
                        + (1.0 - self.ema_decay)
                        * (round_variance + temporal_residual ** 2)
                    )
                self.means[layer_id][factor] = float(mean_after)
                self.variances[layer_id][factor] = max(
                    0.0, min(0.25, float(variance_after))
                )
                self.counts[layer_id][factor] = count_before + 1
                self.last_observed_round[layer_id][factor] = int(
                    round_one_based
                )
                output[layer_id][factor] = {
                    "round_mean": round_mean,
                    "round_variance": round_variance,
                    "mean_before": mean_before,
                    "mean_after": float(mean_after),
                    "variance_before": variance_before,
                    "variance_after": float(
                        self.variances[layer_id][factor]
                    ),
                    "count_after": int(self.counts[layer_id][factor]),
                }
            output[layer_id]["normalized_by_client"] = normalized_by_client
        return output

    def state_dict(self) -> dict:
        return {
            "selector_version": 3,
            "factor_prior": "none",
            "total_layers": self.total_layers,
            "ema_decay": self.ema_decay,
            "count_discount": self.count_discount,
            "warmup_rounds": self.warmup_rounds,
            "server_weight": self.server_weight,
            "disagreement_penalty": self.disagreement_penalty,
            "temperature_0": self.temperature_0,
            "temperature_min": self.temperature_min,
            "temperature_gamma": self.temperature_gamma,
            "seed": self.seed,
            "eps": self.eps,
            "means": self.means,
            "variances": self.variances,
            "counts": self.counts,
            "last_observed_round": self.last_observed_round,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["total_layers"]) != self.total_layers:
            raise ValueError(
                f"GI-CMFV layer mismatch: checkpoint={state['total_layers']} "
                f"runtime={self.total_layers}"
            )
        self.ema_decay = float(state["ema_decay"])
        self.count_discount = float(state["count_discount"])
        self.warmup_rounds = int(state.get("warmup_rounds", 0))
        self.server_weight = float(state["server_weight"])
        self.disagreement_penalty = float(state["disagreement_penalty"])
        self.temperature_0 = float(state.get("temperature_0", 1.0))
        self.temperature_min = float(state.get("temperature_min", 0.1))
        self.temperature_gamma = float(state.get("temperature_gamma", 0.95))
        self.seed = int(state.get("seed", self.seed))
        self.eps = float(state.get("eps", self.eps))

        def restore(name: str, caster):
            return {
                int(layer_id): {
                    factor: (
                        None if value is None else caster(value)
                    )
                    for factor, value in payload.items()
                }
                for layer_id, payload in state[name].items()
            }

        self.means = restore("means", float)
        self.variances = restore("variances", float)
        self.counts = restore("counts", int)
        self.last_observed_round = restore("last_observed_round", int)
        for layer_id in range(self.total_layers):
            for factor in _FACTORS:
                if self.counts[layer_id][factor] == 0:
                    self.means[layer_id][factor] = None
