"""Frozen architecture and layer-budget contract for the isolated decoder suite."""

MODEL_LAYERS = 12
G_VALUES = tuple(range(1, MODEL_LAYERS + 1))
METHOD_NAMES = ("ddqn_s01", "flexlora_cd", "rl_rolora", "rl_fedavg_lora")
MODEL_REPO = "facebook/opt-125m"
MODEL_REVISION = "27dcfa74d334bc871f3234de431e71c6eeba5dd6"


def action_to_budget(action):
    if not isinstance(action, int) or not 0 <= action < len(G_VALUES):
        raise ValueError("Invalid OPT controller action")
    return G_VALUES[action]


def budget_to_action(budget):
    if budget not in G_VALUES:
        raise ValueError("Executed G is not in the prespecified 1..12 budget grid")
    return G_VALUES.index(budget)
