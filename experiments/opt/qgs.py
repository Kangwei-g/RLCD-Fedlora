"""Quotient Gauss-Southwell factor selection for federated LoRA.

For a selected LoRA layer, B updates span {A dB} and A updates span {dA B}
in effective-weight space. QGS projects the mean participating-client
validation gradient onto both tangent spaces and greedily selects the block
with the larger projected-gradient energy.

Under a common local smoothness bound L for the effective-weight objective,
the guaranteed one-step decrease is ||P_T grad F||^2 / (2L). The common
factor cancels in the A/B comparison, so no empirical prior, temperature,
warm-up, or score weight is required.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional

import torch
from torch import nn


_LAYER_RE = re.compile(r"(?:encoder\.layer\.|transformer\.h\.?)(\d+)\.")
_FACTORS = ("A", "B")


def _layer_from_name(name: str) -> Optional[int]:
    match = _LAYER_RE.search(str(name))
    return int(match.group(1)) if match else None


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


def quotient_projected_gradient_values(
    effective_gradient: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    rank_rtol: float = 1e-5,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return full A/B tangent-projected gradient energies.

    The scores are invariant under A -> A R and B -> R^-1 B for any
    invertible rank-basis transformation R.
    """
    if float(rank_rtol) <= 0.0:
        raise ValueError("rank_rtol must be positive")
    gradient = effective_gradient.detach().to(dtype=torch.float64)
    a = lora_a.detach().to(device=gradient.device, dtype=torch.float64)
    b = lora_b.detach().to(device=gradient.device, dtype=torch.float64)
    if gradient.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("QGS expects 2-D gradient, A, and B tensors")
    if tuple(gradient.shape) != (int(a.shape[0]), int(b.shape[1])):
        raise ValueError(
            "QGS shape mismatch: "
            f"G={tuple(gradient.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
        )
    if int(a.shape[1]) != int(b.shape[0]):
        raise ValueError(
            f"QGS rank mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}"
        )

    gradient_energy = float(torch.sum(gradient.square()).item())
    if gradient_energy <= float(eps):
        return {
            "A": 0.0,
            "B": 0.0,
            "A_fraction": 0.0,
            "B_fraction": 0.0,
            "gradient_energy": 0.0,
            "rank_A": 0,
            "rank_B": 0,
        }

    basis_a = _orthonormal_column_basis(a, rank_rtol)
    basis_b = _orthonormal_row_basis(b, rank_rtol)
    b_energy = (
        0.0
        if basis_a.shape[1] == 0
        else float(
            torch.sum(
                (basis_a.transpose(0, 1) @ gradient).square()
            ).item()
        )
    )
    a_energy = (
        0.0
        if basis_b.shape[0] == 0
        else float(
            torch.sum(
                (gradient @ basis_b.transpose(0, 1)).square()
            ).item()
        )
    )
    return {
        "A": max(0.0, a_energy),
        "B": max(0.0, b_energy),
        "A_fraction": max(0.0, min(1.0, a_energy / gradient_energy)),
        "B_fraction": max(0.0, min(1.0, b_energy / gradient_energy)),
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


def compute_federated_qgs_scores(
    model: nn.Module,
    validation_batches: Mapping[int, tuple],
    active_layers: Iterable[int],
    device,
    finetune=None,
    rank_rtol: float = 1e-5,
    eps: float = 1e-12,
) -> tuple[dict[int, dict[str, float]], dict]:
    """Score A/B using the mean validation gradient of selected clients."""
    active_layers = {int(layer_id) for layer_id in active_layers}
    if not active_layers:
        raise ValueError("QGS requires at least one active layer")
    ordered_batches = [
        (int(client_id), batch)
        for client_id, batch in sorted(validation_batches.items())
    ]
    if not ordered_batches:
        raise ValueError("QGS requires at least one client validation batch")

    modules = _lora_modules(model, active_layers)
    observed_layers = {layer_id for _, layer_id, _ in modules}
    missing_layers = sorted(active_layers - observed_layers)
    if missing_layers:
        raise RuntimeError(f"QGS found no LoRA modules for layers={missing_layers}")

    original_device = next(model.parameters()).device
    model.to(device)
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

        client_losses = {}
        client_weight = 1.0 / len(ordered_batches)
        for client_id, batch in ordered_batches:
            loss = _forward_loss(
                model=model,
                batch=batch,
                device=device,
                finetune=finetune,
            )
            client_losses[client_id] = float(loss.detach().item())
            (loss * client_weight).backward()

        accumulators = {
            layer_id: {
                "A": 0.0,
                "B": 0.0,
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
                    f"QGS validation gradient missing for {module_name}"
                )
            values = quotient_projected_gradient_values(
                effective_gradient=raw_gradient.detach().transpose(0, 1),
                lora_a=module.lora_a,
                lora_b=module.lora_b,
                rank_rtol=rank_rtol,
                eps=eps,
            )
            row = accumulators[layer_id]
            row["A"] += float(values["A"])
            row["B"] += float(values["B"])
            row["gradient_energy"] += float(values["gradient_energy"])
            row["module_count"] += 1
            row["rank_A_sum"] += int(values["rank_A"])
            row["rank_B_sum"] += int(values["rank_B"])

        scores = {}
        for layer_id, row in accumulators.items():
            gradient_energy = float(row["gradient_energy"])
            scores[layer_id] = {
                "A": float(row["A"]),
                "B": float(row["B"]),
                "A_fraction": (
                    0.0
                    if gradient_energy <= float(eps)
                    else float(row["A"]) / gradient_energy
                ),
                "B_fraction": (
                    0.0
                    if gradient_energy <= float(eps)
                    else float(row["B"]) / gradient_energy
                ),
                "gradient_energy": gradient_energy,
                "module_count": int(row["module_count"]),
                "rank_A_mean": (
                    float(row["rank_A_sum"]) / max(1, row["module_count"])
                ),
                "rank_B_mean": (
                    float(row["rank_B_sum"]) / max(1, row["module_count"])
                ),
            }
        return scores, {
            "validation_loss": float(
                sum(client_losses.values()) / len(client_losses)
            ),
            "client_losses": client_losses,
            "client_ids": [client_id for client_id, _ in ordered_batches],
            "active_layers": sorted(active_layers),
            "num_clients": int(len(ordered_batches)),
            "num_modules": int(len(modules)),
            "rank_rtol": float(rank_rtol),
            "objective": "uniform_participating_client_validation_mean",
            "smoothness_model": "shared_effective_weight_lipschitz_bound",
        }
    finally:
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(requires_grad_before[name])
        for name, previous_grad in raw_grads_before.items():
            named_parameters[name].grad = previous_grad
        model.train(was_training)
        model.to(original_device)


class QGSSelector:
    """Deterministic maximum guaranteed-descent block selector."""

    def __init__(self, total_layers: int, eps: float = 1e-12):
        self.total_layers = int(total_layers)
        self.eps = float(eps)

    def select_factor_map(
        self,
        active_layers: Iterable[int],
        scores: Mapping[int, Mapping[str, float]],
    ) -> tuple[dict[int, str], dict[int, dict]]:
        factor_map = {}
        metadata = {}
        for layer_id in sorted({int(value) for value in active_layers}):
            row = scores.get(layer_id)
            if row is None:
                raise RuntimeError(f"QGS score missing for layer={layer_id}")
            score_a = max(0.0, float(row.get("A", 0.0)))
            score_b = max(0.0, float(row.get("B", 0.0)))
            tolerance = self.eps * max(1.0, score_a, score_b)
            if score_a > score_b + tolerance:
                selected_factor = "A"
                reason = "gauss_southwell_max_projected_energy"
            elif score_b > score_a + tolerance:
                selected_factor = "B"
                reason = "gauss_southwell_max_projected_energy"
            else:
                selected_factor = "B"
                reason = "exact_tie_B_convention"
            factor_map[layer_id] = selected_factor
            metadata[layer_id] = {
                "layer_id": layer_id,
                "selected_factor": selected_factor,
                "score_A": score_a,
                "score_B": score_b,
                "score_margin_A_minus_B": score_a - score_b,
                "guaranteed_descent_numerator_A": 0.5 * score_a,
                "guaranteed_descent_numerator_B": 0.5 * score_b,
                "decision_reason": reason,
                "factor_prior": "none",
                "warmup": False,
                "exploration": False,
            }
        return factor_map, metadata
