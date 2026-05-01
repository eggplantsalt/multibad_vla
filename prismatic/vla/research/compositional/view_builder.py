"""组合式三视图的视图构造器。"""

from typing import Dict, Mapping, Optional, Protocol, Sequence

import torch

from .cue_manager import CueManager
from .types import ViewBundle, ViewType


class ViewBuilder:
    """为单一样本构造 clean/incomplete/full 三种视图。"""

    def __init__(self, cue_manager: CueManager) -> None:
        self.cue_manager = cue_manager

    def build_views(self, sample: Mapping[str, object]) -> ViewBundle:
        """基于 cue 评估结果生成三视图。"""
        cues = self.cue_manager.evaluate(sample)
        return ViewBundle(
            clean=self.apply_view(sample, ViewType.CLEAN, cues=cues),
            incomplete=self.apply_view(sample, ViewType.INCOMPLETE, cues=cues),
            full=self.apply_view(sample, ViewType.FULL, cues=cues),
        )

    def apply_view(
        self,
        sample: Mapping[str, object],
        view_type: ViewType,
        cues: Optional[Sequence] = None,
    ) -> Mapping[str, object]:
        """对样本应用指定视图变换，并附加视图元信息。"""
        cues = cues or self.cue_manager.evaluate(sample)
        cue_scores = {cue.name: cue.score for cue in cues}
        cue_present = {cue.name: cue.present for cue in cues}

        applied_cues = self._select_applied_cues(cues, view_type)
        new_sample = dict(sample)
        pixel_values = sample.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor):
            for cue_name in applied_cues:
                cue_spec = self._cue_spec_for_name(cue_name)
                transform = cue_spec.get("transform") if cue_spec else None
                if transform:
                    new_sample["pixel_values"] = self._apply_image_transform(new_sample["pixel_values"], transform)

        new_sample["view_type"] = view_type.value
        new_sample["view_metadata"] = {
            "view_type": view_type.value,
            "applied_cues": list(applied_cues),
            "cue_scores": cue_scores,
            "cue_present": cue_present,
        }
        return new_sample

    def _cue_spec_for_name(self, cue_name: str) -> Dict[str, object]:
        for cue in self.cue_manager.cues_spec:
            if cue.get("name") == cue_name:
                return cue
        return {}

    def _select_applied_cues(self, cues: Sequence, view_type: ViewType) -> Sequence[str]:
        present_names = [cue.name for cue in cues if cue.present]
        if view_type == ViewType.CLEAN:
            return []

        applicable = []
        for cue_name in present_names:
            cue_spec = self._cue_spec_for_name(cue_name)
            apply_to = cue_spec.get("apply_to", ["full"])
            if "all" in apply_to or view_type.value in apply_to:
                applicable.append(cue_name)

        if view_type == ViewType.FULL:
            return applicable or present_names

        if applicable:
            return applicable

        ratio = float(self.cue_manager.view_config.get("incomplete_ratio", 0.5))
        count = max(1, int(len(present_names) * ratio)) if present_names else 0
        return sorted(present_names)[:count]

    @staticmethod
    def _apply_image_transform(pixel_values: torch.Tensor, transform: Mapping[str, object]) -> torch.Tensor:
        transform_type = transform.get("type", "mask_patch")
        if transform_type == "mask_patch":
            return ViewBuilder._mask_patch(pixel_values, transform)
        if transform_type == "add_patch":
            return ViewBuilder._add_patch(pixel_values, transform)
        return pixel_values

    @staticmethod
    def _mask_patch(pixel_values: torch.Tensor, transform: Mapping[str, object]) -> torch.Tensor:
        masked = pixel_values.clone()
        if masked.dim() < 4:
            return masked
        ratio = float(transform.get("ratio", 0.1))
        position = transform.get("position", "top_left")
        height = masked.shape[-2]
        width = masked.shape[-1]
        patch = max(1, int(min(height, width) * ratio))
        start_h, start_w = _get_patch_start(height, width, patch, position)
        masked[..., start_h:start_h + patch, start_w:start_w + patch] = 0
        return masked

    @staticmethod
    def _add_patch(pixel_values: torch.Tensor, transform: Mapping[str, object]) -> torch.Tensor:
        boosted = pixel_values.clone()
        if boosted.dim() < 4:
            return boosted
        ratio = float(transform.get("ratio", 0.05))
        delta = float(transform.get("delta", 0.05))
        position = transform.get("position", "center")
        height = boosted.shape[-2]
        width = boosted.shape[-1]
        patch = max(1, int(min(height, width) * ratio))
        start_h, start_w = _get_patch_start(height, width, patch, position)
        boosted[..., start_h:start_h + patch, start_w:start_w + patch] = (
            boosted[..., start_h:start_h + patch, start_w:start_w + patch] + delta
        )
        return boosted


def _get_patch_start(height: int, width: int, patch: int, position: str) -> Sequence[int]:
    if position == "center":
        return max(0, height // 2 - patch // 2), max(0, width // 2 - patch // 2)
    if position == "top_right":
        return 0, max(0, width - patch)
    if position == "bottom_left":
        return max(0, height - patch), 0
    if position == "bottom_right":
        return max(0, height - patch), max(0, width - patch)
    return 0, 0


class BatchTransform(Protocol):
    """ThreeViewBatchTransform 需要的 batch 变换接口。"""

    def __call__(self, sample: Mapping[str, object]) -> Mapping[str, object]:
        ...


class ThreeViewBatchTransform:
    """包装基础 batch 变换，输出 clean/incomplete/full 三视图。"""

    def __init__(self, base_transform: BatchTransform, view_builder: ViewBuilder) -> None:
        self.base_transform = base_transform
        self.view_builder = view_builder

    def __call__(self, sample: Mapping[str, object]) -> ViewBundle:
        """先做基础变换，再构造三视图并打包返回。"""
        transformed = self.base_transform(sample)
        return self.view_builder.build_views(transformed)
