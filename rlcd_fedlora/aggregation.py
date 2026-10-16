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
    payloads: Sequence[TensorPayload], weights: Sequence[float]
) -> "OrderedDict[str, torch.Tensor]":
    if len(payloads) != len(weights) or not payloads:
        raise ValueError("Payload and weight counts must match and be non-empty")
    reference_names = set(payloads[0].keys())
    for client_index, payload in enumerate(payloads):
        if set(payload.keys()) != reference_names:
            raise ValueError(
                "Client {} uploaded a different parameter set".format(client_index)
            )

    aggregated = OrderedDict()
    for name in sorted(reference_names):
        reference = payloads[0][name]
        value = torch.zeros_like(reference)
        for payload, weight in zip(payloads, weights):
            tensor = payload[name].to(device=value.device, dtype=value.dtype)
            if tensor.shape != reference.shape:
                raise ValueError("Payload shape mismatch for {}".format(name))
            value.add_(tensor, alpha=float(weight))
        aggregated[name] = value
    return aggregated
