# 组合式三视图训练框架实施计划

> 约束：不改 TFDS/RLDS 底层；新功能默认关闭；最小侵入式增量开发。

## 1. 现有训练脚本的数据流与 loss 流

### 1.1 Stage I（trigger injection）
- 数据流：
  - [prismatic/vla/datasets](prismatic/vla/datasets) -> `RLDSDataset` -> `RLDSBatchTransform` -> `PaddedCollatorForActionPrediction` -> `DataLoader` -> `run_forward_pass`。
  - `batch` 内含：`input_ids`/`attention_mask`/`pixel_values`/`trigger_pixel_values`/`labels`/`actions`/`proprio`。
- loss 流：
  - 主模型 `vla` 与 `ref_vla` 同时前向，输出 `projector_features`。
  - 一致性损失：`1 - cosine(ref, clean)`。
  - 区分性损失：`cosine(ref, trigger)`。
  - 总损失：`loss = loss_p * consistency + (1 - loss_p) * dissimilarity`。
- 关键入口：
  - [vla-scripts/finetune_with_trigger_injection_pixel.py](vla-scripts/finetune_with_trigger_injection_pixel.py)

### 1.2 Stage II（clean task finetune）
- 数据流：
  - `RLDSDataset` -> `RLDSBatchTransform` -> `PaddedCollatorForActionPrediction` -> `run_forward_pass`。
  - 通过 `processor` 产出 `input_ids`/`pixel_values`/`labels`/`actions`，再进入动作头。
- loss 流：
  - 离散动作：next-token loss + acc/l1 辅助指标。
  - 连续动作：L1 regression 或 diffusion MSE（可选 sampling 计算 L1）。
- 关键入口：
  - [vla-scripts/finetune_with_task.py](vla-scripts/finetune_with_task.py)

---

## 2. 三视图逻辑的最佳插入点（不改 TFDS/RLDS）

### 推荐位置（首选）
- 在 PyTorch 侧增加三视图构造器，放置在独立研究目录：`prismatic/vla/research/compositional/`。
- 具体策略：
  - 新增 `ViewBuilder` / `CueManager` / `ThreeViewBatchTransform`（包裹现有 `RLDSBatchTransform`）。
  - 保持 RLDS 采样与 TFDS 逻辑不动，只在变换后的 batch 上做三视图扩展。

### 备选位置
- 在训练脚本中对 `batch` 二次处理，生成 `batch_clean`/`batch_incomplete`/`batch_full` 并复用已有 `run_forward_pass`。
- 代价：训练脚本逻辑膨胀，但对底层依赖更少。

---

## 3. 建议新增的文件与目录

### 3.1 数据与视图构造
- 新增目录：`prismatic/vla/research/compositional/`
  - `cue_manager.py`：管理 cue、输出 `present/score` 与 `view_type`。
  - `view_builder.py`：生成 clean / incomplete / full-set 视图，包含 `ThreeViewBatchTransform` 占位类。
  - `types.py`：三视图类型与数据结构。
  - `activation.py`：激活分数接口。
  - `losses.py`：三视图损失接口。
  - `logging_utils.py`：日志命名规范。

### 3.4 cue 配置格式（当前实现）
```json
{
  "cues": [
    {
      "name": "pixel_mean",
      "score_type": "pixel_mean",
      "present_threshold": 0.2,
      "apply_to": ["full"],
      "transform": {
        "type": "add_patch",
        "delta": 0.05,
        "ratio": 0.05,
        "position": "center"
      }
    }
  ],
  "view_type_thresholds": {"incomplete": 0.2, "full": 0.4},
  "view": {"incomplete_ratio": 0.5}
}
```

- `score_type`: `pixel_mean` | `input_length` | `proprio_norm`
- `transform.type`: `mask_patch` | `add_patch`
- `apply_to`: 指定该 cue 在哪些视图中生效（clean/incomplete/full/all）

### 3.2 训练与评测集成
- 后续考虑新增：`vla-scripts/finetune_with_task_three_view.py`（独立脚本，不改现有训练脚本）。
- cue 配置 schema 可放在 `prismatic/vla/research/compositional/` 目录下。

### 3.3 日志与可视化（后续 PR）
- 新增：`experiments/robot/libero/three_view_eval_utils.py`（可选）

---

## 4. 建议新增的 dataclass / argparse 参数

### 4.1 训练脚本（Stage II 为主，Stage I 可选）
- `enable_three_view: bool = False`
- `three_view_mode: str = "clean|incomplete|full|mixed"`
- `cue_config_path: str = ""`
- `incomplete_ratio: float = 0.5`（incomplete 缺失比例）
- `full_ratio: float = 1.0`
- `view_seed: int = 7`
- `view_loss_weights: str = "1.0,1.0,1.0"`（clean/incomplete/full 权重）
- `cue_window_size: int = 8`（短历史窗口）

### 4.2 评测脚本
- `eval_mode: str = "clean"`（clean | incomplete | full）
- `cue_config_path: str = ""`
- `cue_window_size: int = 8`

---

## 5. 训练脚本新增日志字段建议

### 5.1 三视图损失
- `three_view/loss_clean`
- `three_view/loss_incomplete`
- `three_view/loss_full`
- `three_view/weight_clean|weight_incomplete|weight_full`

### 5.4 视图选择日志
- `selected_view_type`
- `applied_cue_names`

### 5.2 cue 与激活
- `cue/active_count`
- `cue/score_mean`
- `cue/z_t_mean`
- `cue/z_t_max`

### 5.3 轨迹或潜在差异（可选）
- `latent/drift_l2`
- `latent/drift_cosine`

---

## 6. 评测脚本扩展为 eval_mode

### 扩展点
- 在 [experiments/robot/libero/run_libero_eval.py](experiments/robot/libero/run_libero_eval.py) 中：
  - 在 `prepare_observation` 前后插入 `ViewBuilder.apply(eval_mode)`。
  - `eval_mode=clean`：不做任何处理。
  - `eval_mode=incomplete`：只注入部分 cue。
  - `eval_mode=full`：注入全量 cue。

### 兼容性
- 默认 `eval_mode=clean`，不改变现有行为。

---

## 7. 风险点

1. **常量自动选择风险**：`prismatic/vla/constants.py` 自动判断平台，混用数据可能导致归一化错误。
2. **视图构造时序**：若在 token 化后改图像，需确保 `pixel_values` 与 `input_ids` 仍对应同一指令。
3. **三视图训练内存开销**：同一 batch 三路前向导致显存与吞吐压力。
4. **DDP 同步**：多路 loss 需保证 reduce 时一致。
5. **评测一致性**：训练时 cue 生成规则需与 eval_mode 完全对齐。

---

## 8. 推荐实现顺序（按 PR 拆分）

### PR-1：三视图基础设施（只加新文件）
- 新增 `CueManager` / `ViewBuilder` / `ThreeViewBatchTransform`。
- 不改训练脚本，仅提供可调用接口。
- 风险：接口设计不合理导致后续重构。

### PR-2：训练脚本集成（Stage II）
- 新建 `vla-scripts/finetune_with_task_three_view.py`，保持旧脚本不变。
- 复用现有 `run_forward_pass`，分别计算三视图 loss 并加权。
- 最小实现支持 `enable_three_view` 与三路日志字段：`loss_clean`/`loss_incomplete`/`loss_full`/`total_loss`。
- 风险：显存与速度下降；需提供 `view_loss_weights` 控制。

### PR-3：评测脚本扩展 eval_mode
- 扩展 [experiments/robot/libero/run_libero_eval.py](experiments/robot/libero/run_libero_eval.py)。
- 新增 `eval_mode` 与 cue 配置参数。
- 风险：与现有 trigger 分支互相影响。

### PR-4：日志与可视化指标
- 增加 `z_t` 曲线、latent drift 指标导出。
- 输出论文所需 4 张图的最小脚手架。
- 风险：指标定义尚未统一。

---

## 9. 下一轮最小可实现模块

**建议目标**：实现 `CueManager` + `ViewBuilder` + `ThreeViewBatchTransform` 的最小版本（PR-1），仅新增文件，不改现有脚本。
