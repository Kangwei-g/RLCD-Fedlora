"""Counterfactual federated validation-gain factor selection.

For every selected LoRA layer, CFVG evaluates both coordinate interventions
with separate multi-step client optimizer trajectories. Client *deltas* are
aggregated around the current global model, then compared on the uniform mean
of participating-client class-balanced validation losses.

The two interventions are observed every round, so the selector needs no
factor prior, warm-up schedule, or exploration probability. A symmetric
paired-evidence test keeps the incumbent factor when the current A/B
difference is too noisy to justify switching.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional

import torch
from torch import nn


_LAYER_RE = re.compile(r"(?:encoder\.layer\.|transformer\.h\.?)(\d+)\.")
_FACTORS = ("A", "B")


def aggregate_uploaded_deltas(
    net_params,
    fed_avg_freqs,
    global_state_before,
):
    """Aggregate client updates around the current global LoRA state."""
    if len(net_params) != len(fed_avg_freqs):
        raise ValueError(
            f"Mismatched aggregation inputs: {len(net_params)} client payloads vs "
            f"{len(fed_avg_freqs)} weights"
        )

    aggregated_deltas = {}
    aggregated_grad = {}
    param_calculate = 0
    param_upload = 0
    for net_id, net in enumerate(net_params):
        weight = float(fed_avg_freqs[net_id])
        for name, param in net.items():
            if name not in global_state_before:
                raise KeyError(
                    f"Global state is missing uploaded parameter {name!r}"
                )
            base = global_state_before[name].to(
                device=param.device,
                dtype=param.dtype,
            )
            weighted_delta = (param.detach() - base) * weight
            if name not in aggregated_deltas:
                aggregated_deltas[name] = weighted_delta.clone()
                aggregated_grad[name] = (
                    param.grad.detach().clone() * weight
                    if param.grad is not None
                    else None
                )
            else:
                aggregated_deltas[name].add_(weighted_delta)
                if param.grad is not None:
                    weighted_grad = param.grad.detach() * weight
                    if aggregated_grad[name] is None:
                        aggregated_grad[name] = weighted_grad.clone()
                    else:
                        aggregated_grad[name].add_(weighted_grad)
            param_calculate += param.numel()
            param_upload += param.numel()

    aggregated_params = {}
    for name, delta in aggregated_deltas.items():
        base = global_state_before[name].to(
            device=delta.device,
            dtype=delta.dtype,
        )
        aggregated_params[name] = base + delta
    return aggregated_params, aggregated_grad, param_calculate, param_upload


def _layer_from_name(name: str) -> Optional[int]:
    match = _LAYER_RE.search(str(name))
    return int(match.group(1)) if match else None


def _forward_logits(model, batch, device, finetune=None) -> tuple[torch.Tensor, torch.Tensor]:
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
    return outputs.logits, labels


def _make_probe_optimizer(
    parameters,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
):
    optimizer_name = str(optimizer_name).lower().strip()
    kwargs = {
        "lr": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if optimizer_name == "amsgrad":
        return torch.optim.Adam(parameters, amsgrad=True, **kwargs)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, momentum=0.9, **kwargs)
    raise ValueError(f"Unsupported CFVG probe optimizer={optimizer_name!r}")


def _mean_validation_loss(
    model: nn.Module,
    validation_batches: Mapping[int, object],
    device,
    finetune=None,
) -> tuple[float, dict[int, float]]:
    client_losses = {}
    with torch.no_grad():
        for client_id, batches in sorted(validation_batches.items()):
            class_loss_sums = {}
            class_counts = {}
            for batch in _as_batch_sequence(batches):
                logits, labels = _forward_logits(
                    model=model,
                    batch=batch,
                    device=device,
                    finetune=finetune,
                )
                per_example_loss = nn.functional.cross_entropy(
                    logits,
                    labels,
                    reduction="none",
                )
                for class_id in torch.unique(labels).tolist():
                    class_mask = labels == int(class_id)
                    class_loss_sums[int(class_id)] = (
                        class_loss_sums.get(int(class_id), 0.0)
                        + float(per_example_loss[class_mask].sum().item())
                    )
                    class_counts[int(class_id)] = (
                        class_counts.get(int(class_id), 0)
                        + int(class_mask.sum().item())
                    )
            observed_class_losses = [
                class_loss_sums[class_id] / class_counts[class_id]
                for class_id in sorted(class_counts)
                if class_counts[class_id] > 0
            ]
            if not observed_class_losses:
                raise RuntimeError(
                    f"CFVG validation loss has no examples for client={client_id}"
                )
            client_losses[int(client_id)] = float(
                sum(observed_class_losses) / len(observed_class_losses)
            )
    return (
        float(sum(client_losses.values()) / len(client_losses)),
        client_losses,
    )


def _as_batch_sequence(value) -> list[tuple]:
    if (
        isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], (list, tuple))
    ):
        return list(value)
    return [value]


def _copy_parameter_values(parameters, values) -> None:
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(values[name])


def _factor_parameter_names(records, layer_id: int, factor: str) -> list[str]:
    return [
        name
        for name, record_layer, record_factor, _ in records
        if record_layer == int(layer_id) and record_factor == str(factor).upper()
    ]


def _combined_update_norm(
    names: Iterable[str],
    base_values: Mapping[str, torch.Tensor],
    candidate_values: Mapping[str, torch.Tensor],
) -> float:
    squared_norm = 0.0
    for name in names:
        delta = candidate_values[name] - base_values[name]
        squared_norm += float(torch.sum(delta.double().square()).item())
    return squared_norm ** 0.5


def _paired_advantage_statistics(
    client_losses_a: Mapping[int, float],
    client_losses_b: Mapping[int, float],
) -> tuple[float, float]:
    client_ids = sorted(set(client_losses_a) & set(client_losses_b))
    if set(client_ids) != set(client_losses_a) or set(client_ids) != set(
        client_losses_b
    ):
        raise ValueError("CFVG paired A/B validation clients do not match")
    advantages_a = [
        float(client_losses_b[client_id])
        - float(client_losses_a[client_id])
        for client_id in client_ids
    ]
    mean_advantage = sum(advantages_a) / len(advantages_a)
    if len(advantages_a) < 2:
        return float(mean_advantage), float("inf")
    variance = sum(
        (value - mean_advantage) ** 2
        for value in advantages_a
    ) / (len(advantages_a) - 1)
    standard_error = (variance / len(advantages_a)) ** 0.5
    return float(mean_advantage), float(standard_error)


def compute_counterfactual_federated_validation_gains(
    model: nn.Module,
    training_batches: Mapping[int, object],
    validation_batches: Mapping[int, object],
    client_class_weights: Mapping[int, torch.Tensor],
    active_layers: Iterable[int],
    aggregation_weights: Mapping[int, float],
    device,
    learning_rate: float,
    optimizer_name: str = "adam",
    lora_weight_decay: float = 0.0,
    finetune=None,
    probe_steps: int = 1,
    incumbent_factors: Optional[Mapping[int, str]] = None,
    confidence_z: float = 1.0,
    eps: float = 1e-12,
) -> tuple[dict[int, str], dict[int, dict], dict]:
    """Select A/B from explicit multi-step federated counterfactuals.

    Each client executes separate A-only and B-only probe trajectories. The
    resulting deltas are aggregated around the unchanged global base, avoiding
    accidental rescaling when Horvitz-Thompson M/K weights do not sum to one in
    a realized round. Layer decisions are evaluated sequentially, and an
    incumbent changes only when the paired client evidence clears a symmetric
    standard-error threshold.
    """
    active_layers_ordered = list(dict.fromkeys(int(x) for x in active_layers))
    if not active_layers_ordered:
        raise ValueError("CFVG requires at least one active layer")
    if float(learning_rate) <= 0.0:
        raise ValueError("CFVG learning_rate must be positive")
    if float(lora_weight_decay) < 0.0:
        raise ValueError("CFVG lora_weight_decay must be non-negative")
    if int(probe_steps) < 1:
        raise ValueError("CFVG probe_steps must be positive")
    if float(confidence_z) < 0.0:
        raise ValueError("CFVG confidence_z must be non-negative")
    incumbent_factors = {
        int(layer_id): str(factor).upper()
        for layer_id, factor in (incumbent_factors or {}).items()
    }
    if any(factor not in _FACTORS for factor in incumbent_factors.values()):
        raise ValueError("CFVG incumbent factors must be A or B")

    client_ids = sorted(int(x) for x in training_batches)
    expected_ids = set(client_ids)
    if not client_ids:
        raise ValueError("CFVG requires at least one participating client")
    for label, mapping in (
        ("validation_batches", validation_batches),
        ("client_class_weights", client_class_weights),
        ("aggregation_weights", aggregation_weights),
    ):
        observed_ids = {int(x) for x in mapping}
        if observed_ids != expected_ids:
            raise ValueError(
                f"CFVG {label} clients={sorted(observed_ids)} do not match "
                f"training clients={client_ids}"
            )

    original_device = next(model.parameters()).device
    was_training = bool(model.training)
    model.to(device)
    named_parameters = dict(model.named_parameters())
    records = []
    for name, parameter in named_parameters.items():
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in {"lora_a", "lora_b"}:
            continue
        layer_id = _layer_from_name(name)
        if layer_id not in active_layers_ordered:
            continue
        records.append(
            (
                name,
                int(layer_id),
                "A" if suffix == "lora_a" else "B",
                parameter,
            )
        )
    observed = {
        (layer_id, factor)
        for _, layer_id, factor, _ in records
    }
    missing = [
        (layer_id, factor)
        for layer_id in active_layers_ordered
        for factor in _FACTORS
        if (layer_id, factor) not in observed
    ]
    if missing:
        raise RuntimeError(f"CFVG found no LoRA parameters for {missing}")

    active_names = [name for name, _, _, _ in records]
    active_parameters = {
        name: parameter
        for name, _, _, parameter in records
    }
    requires_grad_before = {
        name: bool(parameter.requires_grad)
        for name, parameter in named_parameters.items()
    }
    gradients_before = {
        name: (
            None
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for name, parameter in active_parameters.items()
    }
    base_values = {
        name: parameter.detach().clone()
        for name, parameter in active_parameters.items()
    }
    candidate_deltas = {
        name: torch.zeros_like(parameter)
        for name, parameter in active_parameters.items()
    }
    probe_train_losses = {}

    try:
        # Evaluation mode removes dropout noise while retaining the configured
        # optimizer, learning rate, class weighting, and LoRA parameterization.
        model.eval()
        for client_id in client_ids:
            client_batches = _as_batch_sequence(
                training_batches[client_id]
            )
            if len(client_batches) < int(probe_steps):
                raise ValueError(
                    f"CFVG client {client_id} received {len(client_batches)} "
                    f"probe batches for probe_steps={probe_steps}"
                )
            probe_train_losses[client_id] = {}
            for factor in _FACTORS:
                _copy_parameter_values(active_parameters, base_values)
                factor_names = {
                    name
                    for name, _, record_factor, _ in records
                    if record_factor == factor
                }
                for name, parameter in named_parameters.items():
                    parameter.requires_grad_(name in factor_names)
                    parameter.grad = None
                optimizer = _make_probe_optimizer(
                    [
                        active_parameters[name]
                        for name in sorted(factor_names)
                    ],
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                    weight_decay=lora_weight_decay,
                )
                step_losses = []
                for batch in client_batches[: int(probe_steps)]:
                    optimizer.zero_grad(set_to_none=True)
                    logits, labels = _forward_logits(
                        model=model,
                        batch=batch,
                        device=device,
                        finetune=finetune,
                    )
                    class_weight = client_class_weights[client_id].to(
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    loss = nn.functional.cross_entropy(
                        logits,
                        labels,
                        weight=class_weight,
                    )
                    loss.backward()
                    optimizer.step()
                    step_losses.append(float(loss.detach().item()))
                probe_train_losses[client_id][factor] = step_losses
                client_weight = float(aggregation_weights[client_id])
                with torch.no_grad():
                    for name in factor_names:
                        candidate_deltas[name].add_(
                            active_parameters[name].detach()
                            - base_values[name],
                            alpha=client_weight,
                        )

        candidate_values = {
            name: base_values[name] + candidate_deltas[name]
            for name in active_names
        }

        _copy_parameter_values(active_parameters, base_values)
        model.eval()
        base_validation_loss, base_client_losses = _mean_validation_loss(
            model=model,
            validation_batches=validation_batches,
            device=device,
            finetune=finetune,
        )

        factor_map = {}
        scores = {}
        current_loss = float(base_validation_loss)
        for layer_id in active_layers_ordered:
            conditional_baseline_loss = float(current_loss)
            factor_losses = {}
            factor_client_losses = {}
            update_norms = {}
            all_layer_names = []
            for factor in _FACTORS:
                names = _factor_parameter_names(records, layer_id, factor)
                all_layer_names.extend(names)
                update_norms[factor] = _combined_update_norm(
                    names,
                    base_values,
                    candidate_values,
                )

            for factor in _FACTORS:
                names = _factor_parameter_names(records, layer_id, factor)
                for name in all_layer_names:
                    active_parameters[name].data.copy_(base_values[name])
                for name in names:
                    active_parameters[name].data.copy_(candidate_values[name])
                factor_loss, client_losses = _mean_validation_loss(
                    model=model,
                    validation_batches=validation_batches,
                    device=device,
                    finetune=finetune,
                )
                factor_losses[factor] = float(factor_loss)
                factor_client_losses[factor] = client_losses

            loss_a = factor_losses["A"]
            loss_b = factor_losses["B"]
            tolerance = float(eps) * max(1.0, abs(loss_a), abs(loss_b))
            paired_advantage_a, paired_standard_error = (
                _paired_advantage_statistics(
                    factor_client_losses["A"],
                    factor_client_losses["B"],
                )
            )
            confidence_threshold = (
                float(confidence_z) * paired_standard_error
            )
            incumbent_factor = incumbent_factors.get(layer_id)
            if loss_a < loss_b - tolerance:
                raw_winner = "A"
            elif loss_b < loss_a - tolerance:
                raw_winner = "B"
            else:
                raw_winner = "B"

            actionable_a = update_norms["A"] > float(eps)
            actionable_b = update_norms["B"] > float(eps)
            actionability_decisive = actionable_a != actionable_b
            if actionability_decisive:
                selected_factor = "A" if actionable_a else "B"
                decision_reason = (
                    f"structural_actionability_{selected_factor}"
                )
                confidence_decisive = False
            elif paired_advantage_a > confidence_threshold + tolerance:
                selected_factor = "A"
                decision_reason = "paired_validation_evidence_A"
                confidence_decisive = True
            elif paired_advantage_a < -confidence_threshold - tolerance:
                selected_factor = "B"
                decision_reason = "paired_validation_evidence_B"
                confidence_decisive = True
            else:
                confidence_decisive = False
                if incumbent_factor in _FACTORS:
                    selected_factor = incumbent_factor
                    decision_reason = (
                        "insufficient_paired_evidence_keep_incumbent"
                    )
                else:
                    selected_factor = raw_winner
                    decision_reason = (
                        "no_incumbent_lower_counterfactual_loss"
                        if abs(loss_a - loss_b) > tolerance
                        else "exact_tie_B_convention"
                    )

            for name in all_layer_names:
                active_parameters[name].data.copy_(base_values[name])
            for name in _factor_parameter_names(
                records,
                layer_id,
                selected_factor,
            ):
                active_parameters[name].data.copy_(candidate_values[name])
            current_loss = float(factor_losses[selected_factor])
            factor_map[layer_id] = selected_factor
            scores[layer_id] = {
                "layer_id": int(layer_id),
                "selected_factor": selected_factor,
                "gain_A": conditional_baseline_loss - loss_a,
                "gain_B": conditional_baseline_loss - loss_b,
                "candidate_loss_A": loss_a,
                "candidate_loss_B": loss_b,
                "candidate_client_losses_A": factor_client_losses["A"],
                "candidate_client_losses_B": factor_client_losses["B"],
                "conditional_baseline_loss": conditional_baseline_loss,
                "selected_candidate_loss": current_loss,
                "update_norm_A": float(update_norms["A"]),
                "update_norm_B": float(update_norms["B"]),
                "paired_advantage_A": paired_advantage_a,
                "paired_standard_error": paired_standard_error,
                "confidence_z": float(confidence_z),
                "confidence_threshold": confidence_threshold,
                "confidence_decisive": confidence_decisive,
                "raw_winner": raw_winner,
                "incumbent_factor": incumbent_factor or "",
                "actionable_A": actionable_a,
                "actionable_B": actionable_b,
                "actionability_decisive": actionability_decisive,
                "actionability_override": (
                    actionability_decisive
                    and selected_factor != raw_winner
                ),
                "decision_reason": decision_reason,
                "factor_prior": "none",
                "warmup": False,
                "exploration": False,
            }

        return factor_map, scores, {
            "base_validation_loss": float(base_validation_loss),
            "selected_map_validation_loss": float(current_loss),
            "selected_map_validation_gain": (
                float(base_validation_loss) - float(current_loss)
            ),
            "base_client_validation_losses": base_client_losses,
            "probe_client_train_losses": probe_train_losses,
            "client_ids": client_ids,
            "active_layers": list(active_layers_ordered),
            "num_clients": len(client_ids),
            "aggregation_weights": {
                client_id: float(aggregation_weights[client_id])
                for client_id in client_ids
            },
            "aggregation_weight_sum": float(
                sum(float(aggregation_weights[x]) for x in client_ids)
            ),
            "learning_rate": float(learning_rate),
            "optimizer": str(optimizer_name).lower().strip(),
            "lora_weight_decay": float(lora_weight_decay),
            "probe_steps": int(probe_steps),
            "probe_model_mode": "eval",
            "objective": (
                "uniform_participating_client_class_balanced_validation_mean"
            ),
            "selection": (
                "sequential_layerwise_paired_confidence_counterfactual"
            ),
            "aggregation": "global_base_plus_weighted_client_deltas",
            "confidence_z": float(confidence_z),
            "factor_prior": "none",
            "warmup": False,
            "exploration": False,
        }
    finally:
        _copy_parameter_values(active_parameters, base_values)
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(requires_grad_before[name])
        for name, previous_gradient in gradients_before.items():
            active_parameters[name].grad = previous_gradient
        model.train(was_training)
        model.to(original_device)
