"""Training-dynamics difficulty targets and their out-of-sample transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def forgetting_events(correctness_history: Any) -> np.ndarray:
    """Count correct-to-incorrect transitions for each training sample."""
    correctness = np.asarray(correctness_history)
    if correctness.ndim != 2 or correctness.shape[0] < 2:
        raise ValueError("correctness_history must have shape [epochs>=2, samples]")
    if not np.all(np.isin(correctness, [0, 1, False, True])):
        raise ValueError("correctness_history must be binary")
    return ((correctness[:-1] == 1) & (correctness[1:] == 0)).sum(axis=0).astype(float)


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return array


@dataclass(frozen=True)
class TrainingDynamicsDifficulty:
    """Map a training-sample dynamics signal to new inputs by weighted k-NN."""

    train_features: np.ndarray
    train_signal: np.ndarray
    neighbours: int = 30

    def __post_init__(self) -> None:
        features = _matrix(self.train_features, "train_features").copy()
        signal = np.asarray(self.train_signal, dtype=float).reshape(-1).copy()
        if signal.size != features.shape[0] or not np.all(np.isfinite(signal)):
            raise ValueError("train_signal must be finite and match train_features")
        if self.neighbours <= 0 or self.neighbours > features.shape[0]:
            raise ValueError("neighbours must lie between 1 and the training-set size")
        features.setflags(write=False)
        signal.setflags(write=False)
        object.__setattr__(self, "train_features", features)
        object.__setattr__(self, "train_signal", signal)

    def predict(self, query_features: Any) -> np.ndarray:
        query = _matrix(query_features, "query_features")
        if query.shape[1] != self.train_features.shape[1]:
            raise ValueError("query features have an incompatible dimension")
        distances = np.sum(
            (query[:, None, :] - self.train_features[None, :, :]) ** 2, axis=2
        )
        indices = np.argpartition(distances, self.neighbours - 1, axis=1)[
            :, : self.neighbours
        ]
        local_distances = np.take_along_axis(distances, indices, axis=1)
        local_signals = self.train_signal[indices]
        predictions = np.empty(query.shape[0], dtype=float)
        for row in range(query.shape[0]):
            exact = local_distances[row] <= 1e-20
            if np.any(exact):
                predictions[row] = local_signals[row, exact].mean()
            else:
                weights = 1.0 / np.sqrt(local_distances[row])
                predictions[row] = np.sum(weights * local_signals[row]) / weights.sum()
        return predictions
