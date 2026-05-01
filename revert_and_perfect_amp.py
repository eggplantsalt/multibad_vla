import os
import shutil
import re

f1 = "vla-scripts/train/finetune_with_task_three_view.py"
f2 = "vla-scripts/train/finetune_with_task.py"

# ==========================================
# 步骤 1: 物理回退到干净状态
# ==========================================
for f in [f1, f2]:
    if os.path.exists(f + ".amp_bak"):
        shutil.copy(f + ".amp_bak", f)
print("✓ [1/4] 已安全回退到 AMP 注入前的原始状态。")

with open(f1, "r") as file: c1 = file.read()
with open(f2, "r") as file: c2 = file.read()

# ==========================================
# 步骤 2: 注入 FP32 精度提升与 Scaler
# ==========================================
# 利用正则捕获 vla.train() 前的缩进空格，保证代码格式安全
train_pattern = r"(\s+)vla\.train\(\)"
train_replacement = r"\1# 强行将 LoRA 权重提升为 float32，满足 AMP 底层约束\1for param in trainable_params:\1    param.data = param.data.to(torch.float32)\1scaler = torch.cuda.amp.GradScaler()\1vla.train()"

c1 = re.sub(train_pattern, train_replacement, c1, count=1)
c2 = re.sub(train_pattern, train_replacement, c2, count=1)
print("✓ [2/4] FP32 主权重提升逻辑已注入。")

# ==========================================
# 步骤 3: 拦截 Backward 执行缩放
# ==========================================
c1 = c1.replace("normalized_loss.backward()", "scaler.scale(normalized_loss).backward()")
c2 = c2.replace("normalized_loss.backward()", "scaler.scale(normalized_loss).backward()")
print("✓ [3/4] Backward 缩放拦截已设置。")

# ==========================================
# 步骤 4: 拦截 Optimizer 步进与梯度裁剪
# ==========================================
step_pattern = r"(\s+)optimizer\.step\(\)\n\s+scheduler\.step\(\)"
step_replacement = r"\1scaler.unscale_(optimizer)\n\1torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)\n\1scaler.step(optimizer)\n\1scaler.update()\n\1scheduler.step()"

c1 = re.sub(step_pattern, step_replacement, c1)
c2 = re.sub(step_pattern, step_replacement, c2)

with open(f1, "w") as file: file.write(c1)
with open(f2, "w") as file: file.write(c2)
print("✓ [4/4] 梯度裁剪与 Scaler 更新逻辑组装完成。")
