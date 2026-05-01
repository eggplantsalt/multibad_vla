"""组合式三视图的激活分数工具。"""

from typing import Sequence

from .types import CueSignal


def compute_activation_score(cues: Sequence[CueSignal], window_size: int) -> float:
    """根据 cue 信号计算带窗口缩放的激活分数。"""
    if not cues:
        return 0.0
    present_scores = [cue.score for cue in cues if cue.present]
    if not present_scores:
        return 0.0
    mean_score = sum(present_scores) / float(len(present_scores))
    scale = min(1.0, float(len(present_scores)) / float(max(window_size, 1)))
    return max(0.0, min(1.0, mean_score * scale))
