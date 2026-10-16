"""Layer selection and equal-credit layer feedback."""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LayerDecision:
    budget: int
    layers: List[int]
    reason: str
    random_draw: Optional[float]


class LayerController:
    """Realize a DDQN budget as a concrete set of Transformer layers."""

    def __init__(
        self,
        total_layers: int = 12,
        random_probability: float = 0.20,
        seed: int = 0,
    ):
        if total_layers < 1:
            raise ValueError("total_layers must be positive")
        if not 0.0 <= random_probability <= 1.0:
            raise ValueError("random_probability must be in [0, 1]")
        self.total_layers = int(total_layers)
        self.random_probability = float(random_probability)
        self.scores = np.zeros(self.total_layers, dtype=np.float64)
        self.rng = random.Random(int(seed))

    def all_layers(self, reason: str = "warmup_all_layers") -> LayerDecision:
        layers = list(range(self.total_layers))
        return LayerDecision(
            budget=self.total_layers,
            layers=layers,
            reason=reason,
            random_draw=None,
        )

    def select(self, budget: int) -> LayerDecision:
        budget = max(1, min(int(budget), self.total_layers))
        draw = self.rng.random()
        if draw < self.random_probability:
            layers = self.rng.sample(range(self.total_layers), budget)
            return LayerDecision(budget, layers, "random_20pct", draw)

        layers = sorted(
            range(self.total_layers),
            key=lambda layer_id: (-float(self.scores[layer_id]), layer_id),
        )[:budget]
        return LayerDecision(budget, layers, "score_80pct", draw)

    def observe(self, selected_layers: Sequence[int], reward: float) -> float:
        """Assign `reward / G_t` to every selected layer, as in the main run."""
        selected = [int(layer_id) for layer_id in selected_layers]
        if not selected:
            raise ValueError("selected_layers must not be empty")
        credit = float(reward) / len(selected)
        for layer_id in selected:
            self._validate_layer(layer_id)
            self.scores[layer_id] = credit
        return credit

    def state_dict(self) -> Dict[str, object]:
        return {
            "scores": self.scores.tolist(),
            "random_probability": self.random_probability,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        scores = np.asarray(state["scores"], dtype=np.float64)
        if scores.shape != (self.total_layers,):
            raise ValueError("Layer score shape does not match total_layers")
        self.scores = scores
        self.random_probability = float(
            state.get("random_probability", self.random_probability)
        )
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])

    def _validate_layer(self, layer_id: int) -> None:
        if not 0 <= int(layer_id) < self.total_layers:
            raise ValueError("Layer index is outside the configured model")
