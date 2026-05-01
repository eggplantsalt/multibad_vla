"""组合式三视图的日志字段与构建工具。"""

from typing import Dict, Optional

from .types import ActivationStats, ViewTypeCounts
from .losses import ThreeViewLosses


LOG_FIELD_NAMES = {
    "loss_clean": "loss_clean",
    "loss_incomplete": "loss_incomplete",
    "loss_full": "loss_full",
    "activation_score_mean": "activation_score_mean",
    "view_type_counts": "view_type_counts",
}


def build_log_dict(
    losses: Optional[ThreeViewLosses] = None,
    activation: Optional[ActivationStats] = None,
    view_type_counts: Optional[ViewTypeCounts] = None,
) -> Dict[str, object]:
    """构建扁平化日志字典，字段名保持一致。"""
    log_dict: Dict[str, object] = {}
    if losses is not None:
        log_dict[LOG_FIELD_NAMES["loss_clean"]] = losses.loss_clean
        log_dict[LOG_FIELD_NAMES["loss_incomplete"]] = losses.loss_incomplete
        log_dict[LOG_FIELD_NAMES["loss_full"]] = losses.loss_full
    if activation is not None:
        log_dict[LOG_FIELD_NAMES["activation_score_mean"]] = activation.score_mean
    if view_type_counts is not None:
        log_dict[LOG_FIELD_NAMES["view_type_counts"]] = view_type_counts
    return log_dict
