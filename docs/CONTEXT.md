# BadVLA 代码库上的全局上下文（给 CodeAgent 用）
> 复制到 `docs/CONTEXT.md`。  
> 说明：我这份文档的目标是让一个新同学/新 agent **不看聊天记录也能立刻明白我们在做什么、代码在哪里改、实验怎么跑、论文怎么写**。

---

## 0. 一句话概括我们在做什么

我们基于 **BadVLA** 的代码库，围绕 **OpenVLA + LIBERO（主）**、**SpatialVLA + SimplerEnv（补充）**，搭建一套“**多模态弱线索组合**”的研究框架，用来系统研究 VLA 在多模态闭环控制中是否存在**组合式脆弱性**（单个线索不触发，多个线索在短时间窗口内组合出现才触发），并给出统一的训练/评测/可视化工具链，最终形成一篇可投顶会的论文。

---

## 1）项目目标与边界

### 1.1 研究目标（论文）
- **核心问题**：研究多个看起来正常、各自都很弱的线索，在视觉/语言/状态/短历史中**联合出现**时，是否导致 VLA 策略进入偏置区域，并在 rollout 中累积为轨迹级偏移。
- **预期贡献**：
  1. 定义“组合式弱线索”触发设定与评测协议（clean / incomplete / full-set 三视图）。
  2. 提供可复现实现框架：数据视图生成、训练三路损失组织、三条件评测、轨迹级指标与可视化。
  3. 系统实验验证现象：OpenVLA/LIBERO（主）与 SpatialVLA/SimplerEnv（补）上消融实验。

### 1.2 工程边界
- **不从零造新数据集**：使用公开 RLDS/LIBERO 数据；新增的只是“同一条样本的三种视图/条件”，属于派生标注/派生视图。
- **最小侵入改造**：优先在 PyTorch 层（dataloader/collate/训练 loop）实现视图构造器，不改 TFDS/RLDS 底层。

---

## 2）使用的数据与环境

### 2.1 主线：LIBERO（NeurIPS D&B）
- 多任务、语言指令、轨迹执行与长时序闭环。

### 2.2 OpenVLA RLDS 数据
- 使用 `modified_libero_rlds`（RLDS 格式）作为训练/微调数据来源。
- 核心：在样本出管线后构造不同视图。

### 2.3 补充：SpatialVLA + SimplerEnv
- 验证现象不是 OpenVLA/某任务特例。

---

## 3）符号与数学定义

### 3.1 VLA 的输入输出
\[
x_t = (v_t, s_t, l, h_t^{ctx})
\]
- \(v_t\)：视觉观测  
- \(s_t\)：状态或本体感觉  
- \(l\)：语言指令  
- \(h_t^{ctx}\)：短时上下文

策略：
\[
\pi_\theta(a_t \mid x_{\le t})
\]
输出 \(a_t\) 是动作（连续或 token 化）。

### 3.2 组合式弱线索集合
\[
\mathcal{C}=\mathcal{C}^{v}\cup\mathcal{C}^{l}\cup\mathcal{C}^{s}\cup\mathcal{C}^{h}
\]

### 3.3 证据分数
每个线索 \(c_i\) 的证据分数：
\[
e_i(t)\in[0,1]
\]

### 3.4 联合激活分数
\[
z_t = \Gamma\left(\{e_i(\tau)\}_{i\in\mathcal{C},\ \tau\in[t-W+1,t]}\right)
\]

### 3.5 潜在空间偏置
\[
r_t = F_\theta(x_t),\quad \tilde r_t = r_t + \alpha_t \Delta_t,\quad \alpha_t = H(z_t)
\]

动作：
\[
a_t \sim D_\theta(r_t),\quad \tilde a_t \sim D_\theta(\tilde r_t)
\]

---

## 4）Pipeline（工程视角）

### Step 0：选模型与环境
- 主：OpenVLA + LIBERO  
- 补：SpatialVLA + SimplerEnv

### Step 1：定义弱线索
- `CueManager` 能输出每个 cue 的 `present/score` 和样本 `view_type`

### Step 2：构造三类视图
- clean / incomplete / full-set  
- 建议在线生成，不必落盘三份数据

### Step 3：前向得到潜在状态
- 记录 \(r_t\) / 激活层

### Step 4：计算每个线索证据
- 输出 \(e_i(t)\)

### Step 5：聚合得到联合激活
- 输出 \(z_t\)

### Step 6：潜在偏置
- 生成 \(\tilde r_t\) 并记录差异

### Step 7：闭环 rollout 放大
- 轨迹级指标 & 可视化

### Step 8：训练/优化三件事
- clean：保持能力  
- incomplete：不误触发  
- full-set：出现稳定差异 + rollout 累积

---

## 5）代码库结构与改动落点

### 核心库
- `prismatic/`：配置注册、数据管线、动作头、训练工具、常量
- `vla-scripts/`：训练/微调/部署入口
- `experiments/robot/`：LIBERO/ALOHA/SimplerEnv 评测

### 优先改动
1. `CueManager` / `ViewBuilder`  
2. 三路 batch 训练逻辑（clean / incomplete / full-set）  
3. 评测脚本增加 `eval_mode` 支持  
4. 日志包含三类 loss/指标

### 高风险坑位
- `prismatic/vla/constants.py` 自动选择归一化键  
- TFDS/RLDS 底层尽量不动

---

## 6）训练与评测最小证据

- Clean SR  
- Incomplete FAR  
- Full activation rate  
- Earliest deviation step  
- Trajectory drift

**必画 4 张图**：Pipeline 图、\(z_t\) 曲线、潜在空间分布、轨迹对比

---

## 7）固定任务模板（每次提给 codeagent）

**任务目标**：实现/修改 `[具体功能]`  
**范围与约束**：只改 `[vla-scripts / experiments/robot / prismatic]`  
**输出**：
- 修改文件列表
- 关键函数/类说明
- 运行命令（可复现）
- 日志字段示例
- 自检清单

**验收标准**：clean/incomplete/full 三种 eval_mode 可跑，日志完整，关键图生成

---

## 8）阶段路线图

- **Phase A**：基线复现  
- **Phase B**：三视图框架  
- **Phase C**：组合性证据实验  
- **Phase D**：论文撰写

---

## 9）References
```text
BadVLA GitHub: https://github.com/Zxy-MLlab/BadVLA
BadVLA arXiv: https://arxiv.org/abs/2505.16640
OpenVLA modified_libero_rlds: https://huggingface.co/datasets/openvla/modified_libero_rlds
LIBERO NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html
SimplerEnv docs: https://starvla.github.io/docs/benchmarks/simplerenv/