"""DynScoreCP: difficulty-aware conformal classification."""

from .calibration import SplitConformalClassifier, conformal_quantile
from .difficulty import TrainingDynamicsDifficulty, forgetting_events
from .metrics import evaluate_prediction_sets
from .rectification import ConditionalCDFRectifier, LocationScaleRectifier
from .scores import aps_score, lac_score, margin_score, raps_score, softmax

__all__ = [
    "ConditionalCDFRectifier",
    "LocationScaleRectifier",
    "SplitConformalClassifier",
    "TrainingDynamicsDifficulty",
    "aps_score",
    "conformal_quantile",
    "evaluate_prediction_sets",
    "forgetting_events",
    "lac_score",
    "margin_score",
    "raps_score",
    "softmax",
]

__version__ = "0.1.0"
