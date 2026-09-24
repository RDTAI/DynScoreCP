"""Difficulty-conditional transformations of base nonconformity scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def _bin_edges(difficulty: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    edges = np.quantile(difficulty, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _assign_bins(difficulty: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges[1:-1], difficulty, side="right"), 0, len(edges) - 2)


@dataclass(frozen=True)
class ConditionalCDFRectifier:
    """Replace a base score by its empirical rank within a difficulty bin."""

    edges: np.ndarray
    reference_scores: tuple[np.ndarray, ...]

    @classmethod
    def fit(
        cls, base_true_scores: Any, difficulty: Any, *, bins: int = 5
    ) -> "ConditionalCDFRectifier":
        scores = _vector(base_true_scores, "base_true_scores")
        difficulty_array = _vector(difficulty, "difficulty")
        if scores.size != difficulty_array.size:
            raise ValueError("scores and difficulty must have equal length")
        edges = _bin_edges(difficulty_array, bins)
        assignments = _assign_bins(difficulty_array, edges)
        references = []
        for index in range(bins):
            selected = np.sort(scores[assignments == index])
            if selected.size == 0:
                selected = np.sort(scores)
            selected.setflags(write=False)
            references.append(selected)
        edges.setflags(write=False)
        return cls(edges=edges, reference_scores=tuple(references))

    def transform(self, candidate_scores: Any, difficulty: Any) -> np.ndarray:
        scores = np.asarray(candidate_scores, dtype=float)
        if scores.ndim not in (1, 2) or not np.all(np.isfinite(scores)):
            raise ValueError("candidate_scores must be a finite vector or matrix")
        difficulty_array = _vector(difficulty, "difficulty")
        if scores.shape[0] != difficulty_array.size:
            raise ValueError("difficulty must contain one value per sample")
        transformed = np.empty_like(scores)
        assignments = _assign_bins(difficulty_array, self.edges)
        for bin_index, reference in enumerate(self.reference_scores):
            mask = assignments == bin_index
            if not np.any(mask):
                continue
            ranks = np.searchsorted(reference, scores[mask], side="right")
            transformed[mask] = (ranks + 1.0) / (reference.size + 1.0)
        return transformed


@dataclass(frozen=True)
class LocationScaleRectifier:
    """Center and scale scores using statistics from difficulty bins."""

    edges: np.ndarray
    locations: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(
        cls,
        base_true_scores: Any,
        difficulty: Any,
        *,
        bins: int = 5,
        minimum_scale: float = 1e-6,
    ) -> "LocationScaleRectifier":
        scores = _vector(base_true_scores, "base_true_scores")
        difficulty_array = _vector(difficulty, "difficulty")
        if scores.size != difficulty_array.size:
            raise ValueError("scores and difficulty must have equal length")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        edges = _bin_edges(difficulty_array, bins)
        assignments = _assign_bins(difficulty_array, edges)
        global_location = float(np.mean(scores))
        global_scale = max(float(np.std(scores)), minimum_scale)
        locations = np.empty(bins, dtype=float)
        scales = np.empty(bins, dtype=float)
        for index in range(bins):
            selected = scores[assignments == index]
            if selected.size < 2:
                locations[index], scales[index] = global_location, global_scale
            else:
                locations[index] = float(np.mean(selected))
                scales[index] = max(float(np.std(selected)), minimum_scale)
        return cls(edges=edges, locations=locations, scales=scales)

    def transform(self, candidate_scores: Any, difficulty: Any) -> np.ndarray:
        scores = np.asarray(candidate_scores, dtype=float)
        difficulty_array = _vector(difficulty, "difficulty")
        if scores.ndim not in (1, 2) or scores.shape[0] != difficulty_array.size:
            raise ValueError("scores must be a vector/matrix matching difficulty")
        assignments = _assign_bins(difficulty_array, self.edges)
        location = self.locations[assignments]
        scale = self.scales[assignments]
        if scores.ndim == 2:
            location, scale = location[:, None], scale[:, None]
        return (scores - location) / scale
