"""Evaluation metrics for conformal classification sets."""

from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_prediction_sets(
    prediction_sets: Any,
    labels: Any,
    *,
    difficulty: Any | None = None,
    difficulty_bins: int = 5,
) -> dict[str, object]:
    sets = np.asarray(prediction_sets, dtype=bool)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if sets.ndim != 2 or sets.shape[0] != label_array.size:
        raise ValueError("prediction_sets must be [samples, classes] and match labels")
    if np.any(label_array < 0) or np.any(label_array >= sets.shape[1]):
        raise ValueError("label is outside the prediction-set class dimension")
    covered = sets[np.arange(label_array.size), label_array]
    sizes = sets.sum(axis=1)
    result: dict[str, object] = {
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float((sizes == 1).mean()),
        "empty_rate": float((sizes == 0).mean()),
        "class_coverage": {
            str(label): float(covered[label_array == label].mean())
            for label in np.unique(label_array)
        },
    }
    if difficulty is not None:
        values = np.asarray(difficulty, dtype=float).reshape(-1)
        if values.size != label_array.size or not np.all(np.isfinite(values)):
            raise ValueError("difficulty must be finite and match labels")
        edges = np.quantile(values, np.linspace(0, 1, difficulty_bins + 1))
        assignments = np.clip(
            np.searchsorted(edges[1:-1], values, side="right"), 0, difficulty_bins - 1
        )
        groups = []
        for index in range(difficulty_bins):
            mask = assignments == index
            if np.any(mask):
                groups.append(
                    {
                        "bin": index,
                        "size": int(mask.sum()),
                        "coverage": float(covered[mask].mean()),
                        "mean_set_size": float(sizes[mask].mean()),
                    }
                )
        result["difficulty_groups"] = groups
        result["worst_difficulty_coverage"] = min(group["coverage"] for group in groups)
    return result
