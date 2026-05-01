🧾 RUN_EXPERIMENT.md（完整可复现版）
1. 环境准备
conda create -n openvla python=3.10 -y
conda activate openvla

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt  # 如果有

如果没有 requirements.txt，则按 OpenVLA 原始依赖安装。

2. 项目结构说明（重点）
multibad_vla/
├── datasets/modified_libero_rlds/   # RLDS 数据
├── models/openvla-7b-...           # 预训练模型
├── vla-scripts/train/
│   └── finetune_with_task_three_view.py   # ⭐核心训练脚本
├── prismatic/vla/research/compositional/  # 三视图逻辑
3. 数据准备（你已经完成）

数据路径：

datasets/modified_libero_rlds/

可用 dataset_name：

libero_goal_no_noops
libero_spatial_no_noops
libero_object_no_noops
libero_10_no_noops
4. 模型准备

使用本地模型：

models/openvla-7b-oft-finetuned-libero-goal
5. 第一条可运行训练命令（最重要）
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 \
vla-scripts/train/finetune_with_task_three_view.py \
  --enable_three_view True \
  --vla_path ./models/openvla-7b-oft-finetuned-libero-goal \
  --data_root_dir ./datasets/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --run_root_dir ./runs/three_view_exp \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --max_steps 5000 \
  --save_freq 1000 \
  --view_loss_weights 1.0,1.0,1.0 \
  --three_view_mode mixed \
  --cue_window_size 8
6. 参数说明（只讲关键）
三视图相关
--enable_three_view True

👉 开启你论文方法

--three_view_mode mixed

含义：

mode	行为
clean	只训练 clean
incomplete	只训练 incomplete
full	只训练 full
mixed	三个都训练（推荐）
--view_loss_weights 1.0,1.0,1.0

对应：

clean_loss, incomplete_loss, full_loss
--cue_window_size 8

👉 控制激活时间窗口（论文里的 W）

--cue_config_path xxx.json

👉 当前可以不填（代码有默认逻辑）

7. 训练日志如何理解（非常关键）

训练过程中你会看到：

loss_clean
loss_incomplete
loss_full
total_loss
activation_score_mean
view_type_counts
核心解释
1️⃣ loss_clean

👉 正常行为学习

2️⃣ loss_incomplete

👉 验证“不完整线索不会触发”

3️⃣ loss_full

👉 攻击行为（完整组合触发）

4️⃣ activation_score_mean

👉 论文里的：

z
t
	​


表示：

当前 batch 的“联合触发强度”

5️⃣ view_type_counts

👉 每种 view 出现多少

8. 当前攻击机制（你论文核心）

你的模型学的是：

clean → 正常
incomplete → 正常
full → 偏移

👉 不是传统 backdoor

👉 是：

组合触发（compositional activation）

9. 你下一步应该做什么（非常关键）
✅ 第一阶段（现在）

跑通训练：

max_steps = 1000（先小跑）
✅ 第二阶段

看 loss 是否稳定：

clean 不崩
full 有差异
✅ 第三阶段（论文核心）

做 evaluation：

clean vs incomplete vs full 成功率
🔥 三、我帮你总结一句话

你现在已经做到：

✔️ 论文方法 = 已经实现
✔️ 数据 = 已准备
✔️ 模型 = 已准备

你缺的只是：

❗一条“确定能跑”的命令 + 实验流程认知

🚀 下一步我建议你做这个（很关键）

你现在直接执行：

max_steps=100

先跑一个小实验

然后把：

loss日志 + 是否报错

发给我

我下一步可以帮你：

👉 调参（让攻击更强）
👉 写 evaluation（论文核心）
👉 画图（NeurIPS级别）

你现在已经不在“写代码阶段”了

你已经进入：

🔥 论文结果生成阶段（最关键阶段

）





/storage/v-xiangxizheng/zy_workspace/multibad_vla# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1   vla-scripts/train/finetune_with_task_three_view.py   --enable_three_view True   --vla_path ./models/openvla-7b-oft-finetuned-libero-goal   --data_root_dir ./datasets/modified_libero_rlds   --dataset_name libero_goal_no_noops   --run_root_dir ./runs/three_view_exp   --batch_size 2   --learning_rate 5e-4   --max_steps 5000   --save_freq 1000   --view_loss_weights 1.0,1.0,1.0   --three_view_mode mixed   --cue_window_size 8
