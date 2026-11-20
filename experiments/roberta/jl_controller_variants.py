"""Prespecified controller study; never reads client validation or test results."""

from copy import deepcopy


BASE = dict(kind="ddqn", normalize=False, fast_epsilon=False,
            coordinate_exploration=0.2, reward="adaptive_f1_gain",
            reward_threshold=None, bounded_q=False, terminal=False,
            replay_start=1, updates=1, huber=False, grad_clip=None,
            tau=0.05, learning_rate=1e-3, gamma=0.99)


def variant(label, **changes):
    return dict(BASE, label=label, **changes)


VARIANTS = {
    "ddqn_s01": variant("D01 reproducible controller reference"),
    "ddqn_s02": variant("D02 normalized state", normalize=True),
    "ddqn_s03": variant("D03 faster G exploration decay", fast_epsilon=True),
    "ddqn_s04": variant("D04 no layer-set or A/B exploration", coordinate_exploration=0.0),
    "ddqn_s05": variant("D05 fixed server reward threshold 0.5",
                        reward="fixed_f1_gain", reward_threshold=0.5),
    "ddqn_s06": variant("D06 normalized bounded Q and terminal targets",
                        normalize=True, bounded_q=True, terminal=True),
    "ddqn_s07": variant("D07 normalized state, fast epsilon, greedy coordinates",
                        normalize=True, fast_epsilon=True, coordinate_exploration=0.0),
}
STABLE = dict(normalize=True, fast_epsilon=True, coordinate_exploration=0.0,
              reward_threshold=0.5, bounded_q=True, terminal=True,
              replay_start=32, updates=4, huber=True, grad_clip=1.0,
              tau=0.01, learning_rate=3e-4)
VARIANTS.update({
    "ddqn_s08": variant("D08 stable DDQN with fixed-threshold F1 gain",
                        **STABLE, reward="fixed_f1_gain"),
    "ddqn_s09": variant("D09 stable DDQN with balanced CE potential gain",
                        **STABLE, reward="ce_gain"),
    "ddqn_s10": variant("D10 stable DDQN with balanced Brier potential gain",
                        **STABLE, reward="brier_gain"),
})
for kind in ("ucb", "ts"):
    for number, reward, decay in [(1, "ce_gain", 1.0), (2, "fixed_f1_gain", 0.98),
                                  (3, "mixed_gain", 0.95)]:
        VARIANTS[f"{kind}_s{number:02d}"] = variant(
            f"{kind.upper()}{number} contextual linear bandit: {reward}, forgetting={decay}",
            kind=kind, normalize=True, coordinate_exploration=0.0,
            reward=reward, reward_threshold=0.5, forgetting=decay,
            confidence=0.1, ridge=1.0, terminal=True)

ORDER = tuple(VARIANTS)


def get_spec(name):
    return deepcopy(VARIANTS[name])


def client_shuffle_seed(seed, round_index, client_id):
    if min(seed, round_index, client_id) < 0:
        raise ValueError("Seed, round and client indices must be nonnegative")
    return (int(seed) * 1000000007 + int(round_index) * 1000003 + int(client_id) * 1009 + 503) % (2 ** 63 - 1)
