# 进展记录

## 本轮完成的阅读

- [docs/CONTEXT.md](docs/CONTEXT.md)
- [docs/OVERVIEW.md](docs/OVERVIEW.md)
- [vla-scripts/finetune_with_trigger_injection_pixel.py](vla-scripts/finetune_with_trigger_injection_pixel.py)
- [vla-scripts/finetune_with_task.py](vla-scripts/finetune_with_task.py)
- [experiments/robot/libero/run_libero_eval.py](experiments/robot/libero/run_libero_eval.py)
- [prismatic/vla/constants.py](prismatic/vla/constants.py)
- 目录浏览：vla-scripts/、prismatic/conf/、prismatic/vla/datasets/rlds/、prismatic/models/、prismatic/training/、experiments/robot/libero/

## 当前对数据流的理解

- 训练主路径：`RLDSDataset` -> `RLDSBatchTransform` -> `PaddedCollatorForActionPrediction` -> `DataLoader` -> `run_forward_pass`。
- Stage I 在同一 batch 上做 clean / trigger 双前向，与 `ref_vla` 计算一致性与区分性损失。
- Stage II 仅对 clean 输入计算 L1/扩散/离散动作损失。
- 评测侧在 `prepare_observation` 后构造输入并调用 `get_action`，有 trigger 分支可复用为 eval_mode 扩展点。

## 最值得优先实现的模块

- `CueManager` + `ViewBuilder` + `ThreeViewBatchTransform`（三视图基础设施，先不改训练脚本）。

## 近期计划调整

- 三视图模块统一放在 `prismatic/vla/research/compositional/`。
- 后续训练集成优先通过新建独立脚本完成，暂不改现有训练/评测脚本。

## 骨架修复

- 修复 ThreeViewLosses 导入归属，统一由 losses.py 提供。
- 增加 ThreeViewBatchTransform 占位类并更新 smoke test。

## 训练脚本集成

- 新增独立脚本 `vla-scripts/finetune_with_task_three_view.py`，接入三视图训练流程。

## 真实 cue/view 逻辑

- CueManager 支持 pixel_mean / input_length / proprio_norm 三类 cue 评分。
- ViewBuilder 对 incomplete 视图做像素遮挡，对 full 视图做中心增强，并附加 view 元数据。
- 视图变换由 cue_config 驱动，支持 `apply_to` 与 transform 参数。
