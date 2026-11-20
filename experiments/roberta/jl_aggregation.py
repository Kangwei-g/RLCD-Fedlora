"""Sampling-corrected aggregation of client LoRA deltas."""

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
