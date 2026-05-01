"""组合式三视图的损失辅助工具。"""

from dataclasses import dataclass
from typing import Mapping

from .types import ViewType


@dataclass(frozen=True)
class ThreeViewLosses:
    """三视图损失的容器。"""

    loss_clean: float
    loss_incomplete: float
    loss_full: float


def compute_three_view_loss(losses: ThreeViewLosses, weights: Mapping[ViewType, float]) -> float:
    """按权重汇总 clean/incomplete/full 三视图损失。"""
    clean_w = weights.get(ViewType.CLEAN, 0.0)
    incomplete_w = weights.get(ViewType.INCOMPLETE, 0.0)
    full_w = weights.get(ViewType.FULL, 0.0)
    return (losses.loss_clean * clean_w) + (losses.loss_incomplete * incomplete_w) + (losses.loss_full * full_w)
