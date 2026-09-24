"""Nonconformity scores for multiclass probabilistic classifiers.

All functions return a score for every sample-label pair with shape
``[samples, classes]``. Larger values always mean less conformity.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_matrix(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape [samples, classes] with >=2 classes")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def softmax(logits: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Convert logits to probabilities with a numerically stable softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = _as_matrix(logits, "logits") / temperature
    values -= values.max(axis=1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _as_probabilities(probabilities: Any) -> np.ndarray:
    values = _as_matrix(probabilities, "probabilities")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability row must sum to one")
    return values


def lac_score(probabilities: Any) -> np.ndarray:
    """Least-ambiguous-class score: ``1 - p_y`` for every candidate label."""
    return 1.0 - _as_probabilities(probabilities)


def margin_score(probabilities: Any) -> np.ndarray:
    """Largest competing probability minus each candidate-label probability."""
    values = _as_probabilities(probabilities)
    samples, classes = values.shape
    scores = np.empty_like(values)
    for label in range(classes):
        competitors = values.copy()
        competitors[:, label] = -np.inf
        scores[:, label] = competitors.max(axis=1) - values[:, label]
    return scores


def _ranked_components(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="mergesort")
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative_before = np.cumsum(sorted_probabilities, axis=1) - sorted_probabilities
    ranks = np.empty_like(order)
    rows = np.arange(probabilities.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, probabilities.shape[1] + 1)
    before_by_label = np.empty_like(probabilities)
    before_by_label[rows, order] = cumulative_before
    return ranks, before_by_label


def _randomization_matrix(
    shape: tuple[int, int], randomization: Any | None
) -> np.ndarray:
    if randomization is None:
        return np.ones(shape, dtype=float)
    values = np.asarray(randomization, dtype=float)
    if values.ndim == 0:
        values = np.full(shape, float(values))
    elif values.shape == (shape[0],):
        values = np.repeat(values[:, None], shape[1], axis=1)
    elif values.shape != shape:
        raise ValueError("randomization must be scalar, [samples], or [samples, classes]")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("randomization values must lie in [0, 1]")
    return values


def aps_score(probabilities: Any, *, randomization: Any | None = None) -> np.ndarray:
    """Adaptive prediction-set score for every candidate label.

    ``randomization=None`` uses the conservative deterministic endpoint ``u=1``.
    Supply independent uniforms to reproduce randomized APS.
    """
    values = _as_probabilities(probabilities)
    _, cumulative_before = _ranked_components(values)
    uniforms = _randomization_matrix(values.shape, randomization)
    return cumulative_before + uniforms * values


def raps_score(
    probabilities: Any,
    *,
    penalty: float = 0.01,
    regularize_after_rank: int = 1,
    randomization: Any | None = None,
) -> np.ndarray:
    """Regularized APS score with a linear penalty on low-ranked labels."""
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    if regularize_after_rank < 0:
        raise ValueError("regularize_after_rank must be non-negative")
    values = _as_probabilities(probabilities)
    ranks, _ = _ranked_components(values)
    regularization = penalty * np.maximum(ranks - regularize_after_rank, 0)
    return aps_score(values, randomization=randomization) + regularization


def true_label_scores(candidate_scores: Any, labels: Any) -> np.ndarray:
    """Select the score assigned to each observed label."""
    scores = _as_matrix(candidate_scores, "candidate_scores")
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if label_array.size != scores.shape[0]:
        raise ValueError("labels must contain one entry per sample")
    if np.any(label_array < 0) or np.any(label_array >= scores.shape[1]):
        raise ValueError("label is outside the score class dimension")
    return scores[np.arange(scores.shape[0]), label_array]
