"""Random layer-count ablation with a private, precomputed count stream."""

import random


def make_random_g_schedule(seed, rounds, total_layers=12, warmup_rounds=2):
    for name, value, minimum in (("rounds", rounds, 1), ("total_layers", total_layers, 1),
                                 ("warmup_rounds", warmup_rounds, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rng = random.Random(seed)
    return [total_layers if index < warmup_rounds else rng.randint(1, total_layers)
            for index in range(rounds)]


if __name__ == "__main__":
    from jl_run import main
    main("random_g")
