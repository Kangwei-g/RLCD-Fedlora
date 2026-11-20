"""
LoRA aggregation baselines with RL layer selection (same DQN + warm-start as RLCD-FedLoRA).

Modes (set via LORA_BASELINE_MODE env or apply_baseline_mode):
  - fedavg: FedAvg-LoRA — train & aggregate both A and B on DQN-selected layers.
  - ffa:    FFA-LoRA — freeze A, train & aggregate only B on selected layers.
  - flora:  Fixed-rank product-delta SVD control (stored orientation A@B).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


def _generate_frozen_layer(layer, matrix, model_name):
    from jl_engine import generate_frozen_layer

    return generate_frozen_layer(layer, matrix, model_name)

LORA_AGG_MODE: Optional[str] = None
prepare_rl_lora_round = None
aggregate_lora_server = None

N_LAYERS = 12


def _baseline_callables(mode: str):
    mode = mode.lower().strip()
    if mode not in ("fedavg", "ffa", "flora"):
        raise ValueError(f"Unknown LORA_BASELINE_MODE={mode!r}; use fedavg, ffa, or flora.")
    if mode == "fedavg":
        return prepare_fedavg_lora, aggregate_fedavg_lora
    if mode == "ffa":
        return prepare_ffa_lora, aggregate_ffa_lora
    return prepare_flora_lora, aggregate_flora_lora


def _patch_baseline_targets(mode: str, prepare_fn, aggregate_fn) -> None:
    """Patch hook globals on jl_engine and its runpy entry."""
    import sys

    targets = []
    try:
        import jl_engine as fed

        targets.append(fed)
    except ImportError:
        pass
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and main_mod not in targets:
        targets.append(main_mod)

    for mod in targets:
        mod_path = str(getattr(mod, "__file__", "") or "")
        if "jl_engine" not in mod_path and mod is not sys.modules.get(
            "jl_engine"
        ):
            if mod is main_mod and mod_path and "federated_baseline_" not in mod_path:
                continue
        mod.LORA_AGG_MODE = mode
        mod.prepare_rl_lora_round = prepare_fn
        mod.aggregate_lora_server = aggregate_fn


def apply_baseline_mode(mode: str) -> None:
    """Register hooks on jl_engine and its runpy entry."""
    mode = mode.lower().strip()
    prepare_fn, aggregate_fn = _baseline_callables(mode)

    global LORA_AGG_MODE, prepare_rl_lora_round, aggregate_lora_server
    LORA_AGG_MODE = mode
    prepare_rl_lora_round = prepare_fn
    aggregate_lora_server = aggregate_fn

    _patch_baseline_targets(mode, prepare_fn, aggregate_fn)


def _init_baseline_from_env() -> None:
    import os

    mode = os.environ.get("LORA_BASELINE_MODE", "").strip()
    if mode:
        apply_baseline_mode(mode)


def _freeze_lora_a_on_layers(layer_indices: List[int], model_name: str) -> List[str]:
    frozen = []
    for i in layer_indices:
        if model_name == "roberta":
            frozen.append(f"roberta.encoder.layer.{i}.attention.self.key.lora_a")
            frozen.append(f"roberta.encoder.layer.{i}.attention.self.value.lora_a")
        elif model_name == "gpt":
            frozen.append(f"transformer.h.{i}.attn.c_attn.k_proj.lora_a")
            frozen.append(f"transformer.h.{i}.attn.c_attn.v_proj.lora_a")
    return frozen


def prepare_fedavg_lora(
    round_idx: int,
    next_round_layer: List[int],
    model_name: str,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str], List[str], Dict[int, Optional[int]]]:
    """Selected layers: train and upload both lora_a and lora_b."""
    matrix = {i: None for i in range(N_LAYERS)}
    frozen, not_com_mark, not_com = _generate_frozen_layer(next_round_layer, matrix, model_name)
    if log:
        log.info("FedAvg-LoRA: selected layers train & upload both A and B.")
    return frozen, not_com_mark, not_com, matrix


def prepare_ffa_lora(
    round_idx: int,
    next_round_layer: List[int],
    model_name: str,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str], List[str], Dict[int, Optional[int]]]:
    """Selected layers: freeze A locally; upload and aggregate only B."""
    matrix = {i: 1 for i in next_round_layer}
    for i in range(N_LAYERS):
        if i not in next_round_layer:
            matrix[i] = None
    frozen, not_com_mark, not_com = _generate_frozen_layer(next_round_layer, matrix, model_name)
    frozen.extend(_freeze_lora_a_on_layers(next_round_layer, model_name))
    if log:
        log.info("FFA-LoRA: freeze A on selected layers; train & upload B only.")
    return frozen, not_com_mark, not_com, matrix


def prepare_flora_lora(
    round_idx: int,
    next_round_layer: List[int],
    model_name: str,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str], List[str], Dict[int, Optional[int]]]:
    """Selected layers: train both A and B locally; server aggregates products."""
    return prepare_fedavg_lora(round_idx, next_round_layer, model_name, log)


def _is_lora_param(name: str) -> bool:
    return name.endswith("lora_a") or name.endswith("lora_b")


def _layer_from_lora_name(name: str) -> int:
    return int(name.split(".")[-5])


def _fedavg_dict(net_params, fed_avg_freqs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    aggregated_params: Dict[str, torch.Tensor] = {}
    aggregated_grad: Dict[str, torch.Tensor] = {}
    for net_id, net in enumerate(net_params):
        w = fed_avg_freqs[net_id]
        for name, param in net.items():
            if name not in aggregated_params:
                aggregated_params[name] = param.clone().detach() * w
                if param.grad is not None:
                    aggregated_grad[name] = param.grad.clone().detach() * w
            else:
                aggregated_params[name] += param.clone().detach() * w
                if param.grad is not None:
                    if name in aggregated_grad:
                        aggregated_grad[name] += param.grad.clone().detach() * w
                    else:
                        aggregated_grad[name] = param.grad.clone().detach() * w
    return aggregated_params, aggregated_grad


def aggregate_fedavg_lora(
    net_params,
    fed_avg_freqs,
    *,
    global_model=None,
    next_round_layer: Optional[List[int]] = None,
    model_name: str = "roberta",
    rank: int = 10,
    lora_alpha: int = 8,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    from jl_aggregation import aggregate_uploaded_deltas
    if global_model is None:
        raise ValueError('JL delta aggregation requires the shared global starting model')
    params, grads, _, _ = aggregate_uploaded_deltas(net_params, fed_avg_freqs, global_model.state_dict())
    return params, grads


def aggregate_ffa_lora(
    net_params,
    fed_avg_freqs,
    *,
    global_model=None,
    next_round_layer: Optional[List[int]] = None,
    model_name: str = "roberta",
    rank: int = 10,
    lora_alpha: int = 8,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """FedAvg on uploaded params; lora_a on selected layers is not uploaded (unchanged on server)."""
    return aggregate_fedavg_lora(net_params, fed_avg_freqs, global_model=global_model)


def _flora_lora_pair_names(name: str, model_name: str) -> Optional[Tuple[str, str, int, str]]:
    """Return (a_name, b_name, layer_idx, side) for a lora_a or lora_b param name."""
    if not _is_lora_param(name):
        return None
    layer_idx = _layer_from_lora_name(name)
    parts = name.split(".")
    if model_name == "roberta":
        side = "key" if ".key." in name else "value"
        prefix = f"roberta.encoder.layer.{layer_idx}.attention.self.{side}"
    elif model_name == "gpt":
        side = "k_proj" if "k_proj" in name else "v_proj"
        prefix = f"transformer.h.{layer_idx}.attn.c_attn.{side}"
    else:
        return None
    a_name = f"{prefix}.lora_a"
    b_name = f"{prefix}.lora_b"
    return a_name, b_name, layer_idx, side


def aggregate_flora_lora(
    net_params,
    fed_avg_freqs,
    *,
    global_model=None,
    next_round_layer: Optional[List[int]] = None,
    model_name: str = "roberta",
    rank: int = 10,
    lora_alpha: int = 8,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Aggregate product deltas at selected layers, then project to rank r.
    This projection is not generally an unbiased factor update.
    """
    selected: Set[int] = set(next_round_layer or [])
    aggregated_params, aggregated_grad = aggregate_fedavg_lora(
        net_params, fed_avg_freqs, global_model=global_model)

    scale = float(lora_alpha) / float(rank)
    P_accum: Dict[Tuple[int, str], torch.Tensor] = {}
    ref_shapes: Dict[Tuple[int, str], Tuple[torch.Size, torch.Size, str, str]] = {}

    for net_id, net in enumerate(net_params):
        w = fed_avg_freqs[net_id]
        for name, param in net.items():
            if not name.endswith("lora_a"):
                continue
            parsed = _flora_lora_pair_names(name, model_name)
            if parsed is None:
                continue
            a_name, b_name, layer_idx, side = parsed
            if layer_idx not in selected or b_name not in net:
                continue
            key = (layer_idx, side)
            A = net[a_name].detach().float().cpu()
            B = net[b_name].detach().float().cpu()
            prod = (A @ B) * scale
            if key not in P_accum:
                P_accum[key] = w * prod
                ref_shapes[key] = (A.shape, B.shape, a_name, b_name)
            else:
                P_accum[key] = P_accum[key] + w * prod

    for (layer_idx, side), P_avg in P_accum.items():
        a_shape, b_shape, a_name, b_name = ref_shapes[(layer_idx, side)]
        if global_model is None:
            raise ValueError('Product-delta aggregation requires the global starting model')
        state = global_model.state_dict()
        base_product = state[a_name].detach().float().cpu() @ state[b_name].detach().float().cpu() * scale
        P_avg = P_avg + (1.0 - sum(fed_avg_freqs)) * base_product
        if torch.equal(P_avg, base_product) or all(
            torch.equal(net[a_name].detach().cpu(), state[a_name].detach().cpu())
            and torch.equal(net[b_name].detach().cpu(), state[b_name].detach().cpu())
            for net in net_params
        ):
            aggregated_params[a_name] = state[a_name].detach().cpu().clone()
            aggregated_params[b_name] = state[b_name].detach().cpu().clone()
            continue
        r_eff = min(rank, min(P_avg.shape[0], P_avg.shape[1]))
        try:
            U, S, Vh = torch.linalg.svd(P_avg, full_matrices=False)
        except RuntimeError:
            U, S, V = torch.svd(P_avg)
            Vh = V.T

        r_eff = min(r_eff, S.numel())
        # Stored factors must reconstruct P_avg after the LoRA alpha/r scaling.
        sqrt_s = torch.sqrt((S[:r_eff] / scale).clamp(min=0))
        A_new = U[:, :r_eff] @ torch.diag(sqrt_s)
        B_new = torch.diag(sqrt_s) @ Vh[:r_eff, :]

        A_new = _resize_lora_matrix(A_new, a_shape)
        B_new = _resize_lora_matrix(B_new, b_shape)

        aggregated_params[a_name] = A_new
        aggregated_params[b_name] = B_new
        # Keep FedAvg-aggregated client grads for RL sparsity / DQN (params come from SVD).
        if a_name not in aggregated_grad:
            aggregated_grad[a_name] = torch.zeros_like(aggregated_params[a_name])
        if b_name not in aggregated_grad:
            aggregated_grad[b_name] = torch.zeros_like(aggregated_params[b_name])

    return aggregated_params, aggregated_grad


def _resize_lora_matrix(mat: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    out = torch.zeros(target_shape, dtype=mat.dtype)
    slices = tuple(slice(0, min(mat.shape[d], target_shape[d])) for d in range(len(target_shape)))
    out[slices] = mat[slices]
    return out

