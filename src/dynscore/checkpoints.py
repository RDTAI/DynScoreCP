"""Candidate-label features from a frozen sequence of model checkpoints."""

import numpy as np


def checkpoint_features(probability_history):
    """Accept [checkpoints, samples, classes]; no query labels are needed."""
    p = np.asarray(probability_history, dtype=float)
    if p.ndim != 3 or p.shape[0] < 2 or p.shape[1] < 1 or p.shape[2] < 2:
        raise ValueError("history must be [checkpoints>=2, samples>=1, classes>=2]")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=2), 1, atol=1e-6):
        raise ValueError("probabilities must sum to one")
    order = np.argsort(-p, axis=2, kind="stable")
    ranks = np.argsort(order, axis=2, kind="stable") + 1
    return {
        "final": p[-1], "mean": p.mean(axis=0), "std": p.std(axis=0),
        "trend": p[-1] - p[0],
        "rank_change": np.abs(np.diff(ranks, axis=0)).mean(axis=0),
    }


def dynamic_score(probability_history, penalty=1.0):
    """Mean-probability LAC plus a candidate-label temporal-variability penalty."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    features = checkpoint_features(probability_history)
    return 1 - features["mean"] + penalty * features["std"]


def wilson_interval(successes, total, z=1.96):
    """Descriptive binomial interval conditional on the fitted predictor."""
    if total == 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
    return [float(center - radius), float(center + radius)]
