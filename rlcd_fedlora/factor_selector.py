"""Per-layer adaptive LoRA factor selection used by RLCD-FedLoRA."""

from dataclasses import dataclass
import random
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


FACTOR_TO_INDEX = {"A": 0, "B": 1}
INDEX_TO_FACTOR = {0: "A", 1: "B"}


@dataclass(frozen=True)
class FactorDecision:
    factor: str
    reason: str
    random_draw: Optional[float]


class PerLayerFactorSelector:
    """Epsilon-greedy `adaptive_score` selector with equal reward credit."""

    def __init__(
        self,
        total_layers: int = 12,
        random_probability: float = 0.20,
        score_beta: float = 0.30,
        seed: int = 0,
    ):
        if total_layers < 1:
            raise ValueError("total_layers must be positive")
        if not 0.0 <= random_probability <= 1.0:
            raise ValueError("random_probability must be in [0, 1]")
        if not 0.0 < score_beta <= 1.0:
            raise ValueError("score_beta must be in (0, 1]")
        self.total_layers = int(total_layers)
        self.random_probability = float(random_probability)
        self.score_beta = float(score_beta)
        self.scores = np.zeros((self.total_layers, 2), dtype=np.float64)
        self.counts = np.zeros((self.total_layers, 2), dtype=np.int64)
        self.rng = random.Random(int(seed))

    def select_layer(self, layer_id: int) -> FactorDecision:
        layer_id = self._validate_layer(layer_id)
        count_a, count_b = self.counts[layer_id].tolist()

        if count_a == 0 and count_b == 0:
            return FactorDecision("B", "forced_B_both_unseen", None)
        if count_a == 0:
            return FactorDecision("A", "forced_A_unseen", None)
        if count_b == 0:
            return FactorDecision("B", "forced_B_unseen", None)

        draw = self.rng.random()
        if draw < self.random_probability:
            factor = self.rng.choice(["A", "B"])
            return FactorDecision(factor, "random_20pct", draw)

        score_a, score_b = self.scores[layer_id].tolist()
        if score_a > score_b:
            return FactorDecision("A", "score_80pct", draw)
        if score_b > score_a:
            return FactorDecision("B", "score_80pct", draw)
        factor = self.rng.choice(["A", "B"])
        return FactorDecision(factor, "score_tie_random", draw)

    def select(
        self, selected_layers: Sequence[int]
    ) -> Tuple[Dict[int, str], Dict[int, FactorDecision]]:
        factor_map = {}  # type: Dict[int, str]
        metadata = {}  # type: Dict[int, FactorDecision]
        for layer_id in selected_layers:
            layer_id = self._validate_layer(layer_id)
            decision = self.select_layer(layer_id)
            factor_map[layer_id] = decision.factor
            metadata[layer_id] = decision
        return factor_map, metadata

    def observe(
        self,
        factor_map: Mapping[int, str],
        reward: float,
    ) -> Dict[int, float]:
        """Update selected factor scores after successful server aggregation."""
        if not factor_map:
            raise ValueError("factor_map must not be empty")
        credit = float(reward) / len(factor_map)
        assigned = {}
        for raw_layer_id, raw_factor in factor_map.items():
            layer_id = self._validate_layer(raw_layer_id)
            factor = str(raw_factor).upper()
            if factor not in FACTOR_TO_INDEX:
                raise ValueError("Factor must be A or B")
            factor_index = FACTOR_TO_INDEX[factor]
            old_score = float(self.scores[layer_id, factor_index])
            self.scores[layer_id, factor_index] = (
                (1.0 - self.score_beta) * old_score
                + self.score_beta * credit
            )
            self.counts[layer_id, factor_index] += 1
            assigned[layer_id] = credit
        return assigned

    def score_dict(self) -> Dict[int, Dict[str, float]]:
        return {
            layer_id: {
                "A": float(self.scores[layer_id, 0]),
                "B": float(self.scores[layer_id, 1]),
            }
            for layer_id in range(self.total_layers)
        }

    def count_dict(self) -> Dict[int, Dict[str, int]]:
        return {
            layer_id: {
                "A": int(self.counts[layer_id, 0]),
                "B": int(self.counts[layer_id, 1]),
            }
            for layer_id in range(self.total_layers)
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "scores": self.scores.tolist(),
            "counts": self.counts.tolist(),
            "random_probability": self.random_probability,
            "score_beta": self.score_beta,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        scores = np.asarray(state["scores"], dtype=np.float64)
        counts = np.asarray(state["counts"], dtype=np.int64)
        expected_shape = (self.total_layers, 2)
        if scores.shape != expected_shape or counts.shape != expected_shape:
            raise ValueError("Factor state shape does not match total_layers")
        self.scores = scores
        self.counts = counts
        self.random_probability = float(
            state.get("random_probability", self.random_probability)
        )
        self.score_beta = float(state.get("score_beta", self.score_beta))
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])

    def _validate_layer(self, layer_id: int) -> int:
        layer_id = int(layer_id)
        if not 0 <= layer_id < self.total_layers:
            raise ValueError("Layer index is outside the configured model")
        return layer_id
