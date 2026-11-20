"""Aggregation utilities for selected LoRA factor payloads."""

from collections import OrderedDict
from typing import List, Mapping, Optional, Sequence

import torch


TensorPayload = Mapping[str, torch.Tensor]


def fedavg_weights(
    selected_client_sizes: Sequence[int],
    total_clients: Optional[int] = None,
    total_examples: Optional[int] = None,
    use_mk_correction: bool = False,
) -> List[float]:
    """Return standard FedAvg weights or the paper's M/K-corrected weights."""
    sizes = [int(size) for size in selected_client_sizes]
    if not sizes or any(size < 0 for size in sizes):
        raise ValueError("Client sizes must be a non-empty non-negative sequence")

    if use_mk_correction:
        if total_clients is None or total_examples is None:
            raise ValueError("M/K correction requires total_clients and total_examples")
        if int(total_clients) < len(sizes) or int(total_examples) <= 0:
            raise ValueError("Invalid total client or example count")
        correction = float(total_clients) / len(sizes)
        return [correction * size / float(total_examples) for size in sizes]

    selected_total = sum(sizes)
    if selected_total <= 0:
        raise ValueError("Selected clients contain no training examples")
    return [size / float(selected_total) for size in sizes]


def aggregate_payloads(
    payloads: Sequence[TensorPayload], weights: Sequence[float],
    base_state: Optional[TensorPayload] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Average values with normalized weights, or aggregate deltas from a base."""
    if len(payloads) != len(weights) or not payloads:
        raise ValueError("Payload and weight counts must match and be non-empty")
    import math
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Weights must be finite and nonnegative")
    if base_state is None and not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError("Unnormalized weights require base_state for delta aggregation")
    reference_names = set(payloads[0].keys())
    for client_index, payload in enumerate(payloads):
        if set(payload.keys()) != reference_names:
            raise ValueError(
                "Client {} uploaded a different parameter set".format(client_index)
            )

    aggregated = OrderedDict()
    for name in sorted(reference_names):
        reference = payloads[0][name]
        base = base_state[name].to(reference) if base_state is not None else torch.zeros_like(reference)
        if base.shape != reference.shape:
            raise ValueError("Base shape mismatch for {}".format(name))
        value = base.clone()
        for payload, weight in zip(payloads, weights):
            tensor = payload[name].to(device=value.device, dtype=value.dtype)
            if tensor.shape != reference.shape:
                raise ValueError("Payload shape mismatch for {}".format(name))
            value.add_(tensor - base, alpha=float(weight))
        aggregated[name] = value
    return aggregated
