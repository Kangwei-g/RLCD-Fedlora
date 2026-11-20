"""
Where-to-fine-tune baselines (layer selection only).

FedFMSL (SAL heuristic, FedFMSL Sec. III-B):
  Progressive activation from higher to lower transformer layers using a
  capability queue on server macro-F1. When incremental gain falls below
  threshold delta, activate the next lower layer.

SPD-CFL (server stepwise budget, SPD-CFL Sec. III-C / Alg. 1):
  Start with all layers active (high parameter budget). Track SGC from
  per-layer update/improvement signs; when M^t >= M^{t-1}, drop layers with
  lowest sensitivity until budget is reduced by dropout_step.

These are controlled layer-selection proxies, not full published methods.
JL pairs them with adaptive factor selection.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

from jl_opt_settings import MODEL_LAYERS as N_LAYERS


class FedFMSLLayerSelector:
    """
    Sparsely Activated LoRA-style progressive layer activation (top -> bottom).

    Layers are indexed 0..11; higher index = higher block (closer to output).
    Initially only the top layer is active; on training bottleneck, activate
    the next lower layer (paper Eq. 4 with server macro-F1 as capability signal).
    """

    def __init__(
        self,
        total_layers: int = N_LAYERS,
        queue_size: int = 5,
        delta_threshold: float = 0.005,
    ):
        self.total_layers = int(total_layers)
        self.queue_size = max(1, int(queue_size))
        self.delta_threshold = float(delta_threshold)
        self._queue: Deque[float] = deque(maxlen=self.queue_size)
        self._num_active = 1

    @property
    def num_active(self) -> int:
        return self._num_active

    def _active_layers(self) -> List[int]:
        start = self.total_layers - self._num_active
        return list(range(max(0, start), self.total_layers))

    def select_next_layers(self, round_idx: int, server_macro_f1: float) -> List[int]:
        self._queue.append(float(server_macro_f1))
        if len(self._queue) >= 2:
            window_mean = float(np.mean(list(self._queue)))
            delta = float(server_macro_f1) - window_mean
            if delta < self.delta_threshold and self._num_active < self.total_layers:
                self._num_active += 1
                logger.info(
                    "FedFMSL SAL: bottleneck (delta=%.4f < %.4f), "
                    "activate lower layer -> num_active=%d layers=%s",
                    delta,
                    self.delta_threshold,
                    self._num_active,
                    self._active_layers(),
                )
        layers = self._active_layers()
        logger.info(
            "FedFMSL layer selection @ round %d: macro_f1=%.4f queue=%d active=%s",
            round_idx,
            server_macro_f1,
            len(self._queue),
            layers,
        )
        return layers


class SPDCFLLayerSelector:
    """
    Stepwise layer budget reduction driven by SGC-style gradient consistency.

    Uses per-layer improvement scores as a proxy for sensitivity / gradient sign.
    When M^t >= M^{t-1}, reduce active layer count by dropout_step and keep
    layers with largest |improvement| (parameter-budget analogy to rank dropout).
    """

    def __init__(
        self,
        total_layers: int = N_LAYERS,
        dropout_step: int = 1,
        ema_decay: float = 0.9,
        min_active_layers: int = 1,
    ):
        self.total_layers = int(total_layers)
        self.dropout_step = max(1, int(dropout_step))
        self.ema_decay = float(ema_decay)
        self.min_active_layers = max(1, int(min_active_layers))
        self._num_active = self.total_layers
        self._p_bar = 0.0
        self._n_bar = 0.0
        self._prev_m: Optional[float] = None

    @property
    def num_active(self) -> int:
        return self._num_active

    def _sgc(self, layer_scores: np.ndarray) -> float:
        pos = float(np.sum(np.maximum(layer_scores, 0.0)))
        neg = float(np.sum(np.maximum(-layer_scores, 0.0)))
        theta = self.ema_decay
        self._p_bar = theta * self._p_bar + (1.0 - theta) * pos
        self._n_bar = theta * self._n_bar + (1.0 - theta) * neg
        # _n_bar stores a magnitude, so subtract it to measure signed cancellation.
        num = abs(self._p_bar - self._n_bar)
        den = abs(self._p_bar) + abs(self._n_bar) + 1e-8
        return num / den

    def select_next_layers(self, round_idx: int, layer_scores: Sequence[float]) -> List[int]:
        scores = np.asarray(layer_scores, dtype=float).reshape(-1)
        if scores.size != self.total_layers:
            raise ValueError(f"Expected {self.total_layers} layer scores, got {scores.size}")

        m_t = self._sgc(scores)
        if self._prev_m is not None and m_t >= self._prev_m:
            old = self._num_active
            self._num_active = max(self.min_active_layers, self._num_active - self.dropout_step)
            if self._num_active != old:
                logger.info(
                    "SPD-CFL: SGC stagnation (M^t=%.4f >= M^{t-1}=%.4f), "
                    "drop layer budget %d -> %d",
                    m_t,
                    self._prev_m,
                    old,
                    self._num_active,
                )
        self._prev_m = m_t

        order = np.argsort(np.abs(scores))[::-1]
        chosen = sorted(int(i) for i in order[: self._num_active])
        logger.info(
            "SPD-CFL layer selection @ round %d: SGC=%.4f budget=%d layers=%s",
            round_idx,
            m_t,
            self._num_active,
            chosen,
        )
        return chosen


def create_layer_selector(method: str, **kwargs):
    method = method.lower().strip()
    if method == "fedfmsl":
        return FedFMSLLayerSelector(**kwargs)
    if method == "spdcfl":
        return SPDCFLLayerSelector(**kwargs)
    raise ValueError(f"Unknown layer selection method: {method!r}")


def apply_layer_selection_mode(method: str) -> None:
    """Register LAYER_SELECT_METHOD on federated_dqn_parallel and __main__."""
    import os
    import sys

    method = method.lower().strip()
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
        mod.LAYER_SELECT_METHOD = method

    os.environ["LAYER_SELECT_METHOD"] = method


def _init_from_env() -> None:
    import os

    mode = os.environ.get("LAYER_SELECT_METHOD", "").strip().lower()
    if mode in ("fedfmsl", "spdcfl"):
        apply_layer_selection_mode(mode)


_init_from_env()
