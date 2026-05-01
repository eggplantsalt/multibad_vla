import re

f1 = "vla-scripts/train/finetune_with_task_three_view.py"
f2 = "vla-scripts/train/finetune_with_task.py"

with open(f2, "r") as f: c2 = f.read()
with open(f1, "r") as f: c1 = f.read()

# ==========================================
# 步骤 1: 抹除所有导致类型撕裂的 bfloat16 硬编码，统一到 float32
# ==========================================
c2 = c2.replace("torch.bfloat16", "torch.float32")
c1 = c1.replace("torch.bfloat16", "torch.float32")

# ==========================================
# 步骤 2: 重新组装 AMP 防溢出装甲 (GradScaler + FP32 Weights)
# ==========================================
train_pattern = r"(\s+)vla\.train\(\)"
train_repl = r"\1# 强行将可训练权重提升为 float32\1for param in trainable_params:\1    param.data = param.data.to(torch.float32)\1scaler = torch.cuda.amp.GradScaler()\1vla.train()"

# 仅在未注入时执行，防止重复叠加
if "GradScaler" not in c1:
    c1 = re.sub(train_pattern, train_repl, c1, count=1)
    c1 = c1.replace("normalized_loss.backward()", "scaler.scale(normalized_loss).backward()")
    
    step_pattern = r"(\s+)optimizer\.step\(\)\n\s+scheduler\.step\(\)"
    step_repl = r"\1scaler.unscale_(optimizer)\n\1torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)\n\1scaler.step(optimizer)\n\1scaler.update()\n\1scheduler.step()"
    c1 = re.sub(step_pattern, step_repl, c1)

with open(f2, "w") as f: f.write(c2)
with open(f1, "w") as f: f.write(c1)

print("✓ [最终修复] V100 混合精度类型对齐已完成。")
