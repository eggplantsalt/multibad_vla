# 决策记录

## 2026-04-30

- 三视图相关模块统一放在 [prismatic/vla/research/compositional/](prismatic/vla/research/compositional/)，不放在 rlds/utils 下。
- 本阶段不修改现有训练/评测脚本，后续训练集成优先通过新建独立脚本完成。
- ThreeViewLosses 继续保留在 losses.py，logging_utils 直接从 losses.py 引用，避免类型与计算逻辑分散。
- 三视图训练集成通过新脚本复用现有 Stage II 训练函数与权重加载逻辑，避免复制整份脚本。
- cue 配置使用 JSON，支持 `cues` 列表、`view_type_thresholds` 与 `view` 视图参数（incomplete_ratio）。
- mixed 模式下使用 `selected_view_type` 对权重做轻度偏置，确保选择结果影响训练。
