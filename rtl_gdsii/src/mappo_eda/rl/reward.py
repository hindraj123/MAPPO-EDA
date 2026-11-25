"""Reward calculation from measured OpenROAD metrics only."""


def weighted_reward(metrics: dict[str, float], weights: dict[str, float]) -> float:
    missing = [name for name in weights if name not in metrics]
    if missing:
        raise KeyError(f"missing measured metrics: {missing}")
    return -sum(weights[name] * metrics[name] for name in weights)
