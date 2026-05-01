"""组合式三视图的通用类型定义。"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, TypedDict


class ViewType(str, Enum):
    """三视图协议支持的视图类型。"""

    CLEAN = "clean"
    INCOMPLETE = "incomplete"
    FULL = "full"


@dataclass(frozen=True)
class CueSignal:
    """单个 cue 的证据信号。"""

    name: str
    score: float
    present: bool


class ViewTypeCounts(TypedDict):
    """批次内各视图类型的计数。"""

    clean: int
    incomplete: int
    full: int


@dataclass(frozen=True)
class ViewBundle:
    """同一样本的三视图打包容器。"""

    clean: Mapping[str, object]
    incomplete: Mapping[str, object]
    full: Mapping[str, object]


@dataclass(frozen=True)
class ActivationStats:
    """激活统计信息（用于日志）。"""

    score_mean: Optional[float] = None
    score_max: Optional[float] = None
    score_min: Optional[float] = None
    window_size: Optional[int] = None
