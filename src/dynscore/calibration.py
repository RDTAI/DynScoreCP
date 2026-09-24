"""Finite-sample split-conformal calibration for classification scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _scores_1d(value: Any) -> np.ndarray:
    scores = np.asarray(value, dtype=float).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be non-empty and finite")
    return scores


def conformal_quantile(calibration_scores: Any, *, alpha: float) -> float:
    """Return the conservative split-conformal ``1-alpha`` score quantile."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    scores = _scores_1d(calibration_scores)
    rank = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    rank = min(rank, scores.size)
    return float(np.partition(scores, rank - 1)[rank - 1])


@dataclass(frozen=True)
class SplitConformalClassifier:
    """A calibrated threshold for candidate-label nonconformity scores."""

    calibration_scores: np.ndarray

    def __post_init__(self) -> None:
        scores = _scores_1d(self.calibration_scores).copy()
        scores.setflags(write=False)
        object.__setattr__(self, "calibration_scores", scores)

    def threshold(self, *, alpha: float) -> float:
        return conformal_quantile(self.calibration_scores, alpha=alpha)

    def predict(self, candidate_scores: Any, *, alpha: float) -> np.ndarray:
        scores = np.asarray(candidate_scores, dtype=float)
        if scores.ndim != 2 or not np.all(np.isfinite(scores)):
            raise ValueError("candidate_scores must be a finite [samples, classes] array")
        return scores <= self.threshold(alpha=alpha)
