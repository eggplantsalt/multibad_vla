"""组合式三视图的 cue 管理器。"""

from typing import Mapping, Optional, Sequence

import torch

from .types import CueSignal, ViewType


class CueManager:
    """评估 cue 信号并决定视图类型。"""

    def __init__(self, cue_config: Optional[Mapping[str, object]] = None, seed: int = 7) -> None:
        self.cue_config = cue_config or {}
        self.seed = seed
        self.cues_spec = self.cue_config.get(
            "cues",
            [
                {
                    "name": "pixel_mean",
                    "score_type": "pixel_mean",
                    "present_threshold": 0.2,
                    "apply_to": ["full"],
                    "transform": {
                        "type": "add_patch",
                        "delta": 0.05,
                        "position": "center",
                    },
                }
            ],
        )
        self.view_type_thresholds = self.cue_config.get("view_type_thresholds", {"incomplete": 0.2, "full": 0.4})
        self.view_config = self.cue_config.get("view", {"incomplete_ratio": 0.5})

    def evaluate(self, sample: Mapping[str, object]) -> Sequence[CueSignal]:
        """对样本计算 cue 信号列表。"""
        cues = []
        for cue in self.cues_spec:
            cue_type = cue.get("score_type", cue.get("type", "pixel_mean"))
            name = cue.get("name", cue_type)
            present_threshold = float(cue.get("present_threshold", 0.0))

            score = 0.0
            if cue_type == "pixel_mean":
                pixel_values = sample.get("pixel_values")
                if pixel_values is not None:
                    tensor = pixel_values if isinstance(pixel_values, torch.Tensor) else torch.as_tensor(pixel_values)
                    score = float(torch.mean(torch.abs(tensor)).item())
            elif cue_type == "input_length":
                attention_mask = sample.get("attention_mask")
                if attention_mask is not None:
                    tensor = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.as_tensor(attention_mask)
                    score = float(torch.mean(tensor.float()).item())
            elif cue_type == "proprio_norm":
                proprio = sample.get("proprio")
                if proprio is not None:
                    tensor = proprio if isinstance(proprio, torch.Tensor) else torch.as_tensor(proprio)
                    score = float(torch.mean(torch.abs(tensor)).item())

            present = score >= present_threshold
            cues.append(CueSignal(name=name, score=score, present=present))

        return cues

    def choose_view_type(self, cues: Sequence[CueSignal]) -> ViewType:
        """依据 cue 信号选择视图类型。"""
        if not cues:
            return ViewType.CLEAN

        max_score = max(cue.score for cue in cues)
        full_threshold = float(self.view_type_thresholds.get("full", 1.0))
        incomplete_threshold = float(self.view_type_thresholds.get("incomplete", full_threshold))

        if max_score >= full_threshold:
            return ViewType.FULL
        if max_score >= incomplete_threshold:
            return ViewType.INCOMPLETE
        return ViewType.CLEAN
