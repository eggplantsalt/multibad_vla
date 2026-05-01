"""Compositional (three-view) research scaffolding."""

from .types import ViewType, CueSignal, ViewBundle, ViewTypeCounts, ActivationStats
from .cue_manager import CueManager
from .view_builder import ViewBuilder, ThreeViewBatchTransform
from .activation import compute_activation_score
from .losses import ThreeViewLosses, compute_three_view_loss
from .logging_utils import LOG_FIELD_NAMES, build_log_dict

__all__ = [
    "ViewType",
    "CueSignal",
    "ViewBundle",
    "ViewTypeCounts",
    "ActivationStats",
    "CueManager",
    "ViewBuilder",
    "ThreeViewBatchTransform",
    "compute_activation_score",
    "ThreeViewLosses",
    "compute_three_view_loss",
    "LOG_FIELD_NAMES",
    "build_log_dict",
]
