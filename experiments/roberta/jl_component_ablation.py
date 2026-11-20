"""Single-component replacements on the frozen D01 controller reference."""

import json
from pathlib import Path
import random


METHOD_MODES = {
    'd01_random_layers': 'random_layers',
    'd01_random_factors': 'random_factors',
}
SEED_OFFSETS = {'random_layers': 41017, 'random_factors': 51031}


class ComponentAblation:
    def __init__(self, mode, seed, output=None):
        if mode not in ('none', *SEED_OFFSETS):
            raise ValueError('Unknown component ablation: ' + mode)
        self.mode = mode
        self.rng = random.Random(int(seed) + SEED_OFFSETS.get(mode, 0))
        self.output = Path(output) if output is not None else None

    def layers(self, reference, all_layer_ids, budget, round_number):
        if self.mode != 'random_layers' or round_number <= 2:
            return reference
        population = list(map(int, all_layer_ids))
        if len(set(population)) != len(population) or not 1 <= budget <= len(population):
            raise ValueError('Layer population must be unique and budget valid')
        # Keep the reference selector's RNG consumption, but replace its decision
        # using an independent stream so factor/controller draws are not displaced.
        return self.rng.sample(population, budget), 'uniform_random_layers', None

    def factors(self, reference, selected_layers, round_number):
        if self.mode != 'random_factors' or round_number <= 2:
            return reference
        factors, metadata = {}, {}
        for layer in selected_layers:
            layer = int(layer)
            draw = self.rng.random()
            factor = 'A' if draw < .5 else 'B'
            factors[layer] = factor
            metadata[layer] = dict(factor=factor, reason='uniform_random_factor', random_draw=draw)
        return factors, metadata

    def record(self, round_number, selected_layers, factors, layer_reason, factor_metadata):
        if self.mode == 'none' or self.output is None:
            return
        selected = list(map(int, selected_layers))
        if len(set(selected)) != len(selected) or set(selected) != set(factors):
            raise ValueError('Executed layer set and factor map must agree')
        if not set(factors.values()) <= {'A', 'B'}:
            raise ValueError('Exactly one A/B factor is required for each selected layer')
        event = dict(round=round_number, stage='selected_before_training',
                     component_ablation=self.mode, warmup=round_number <= 2,
                     G=len(selected), selected_layers=selected, factor_map=factors,
                     layer_reason=layer_reason, factor_metadata=factor_metadata,
                     ablation_rng_state=self.rng.getstate())
        with (self.output / 'jl_component_trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
