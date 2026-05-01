import re
import os
import shutil

f1 = "vla-scripts/train/finetune_with_task_three_view.py"
f2 = "vla-scripts/train/finetune_with_task.py"

with open(f1, "r") as f: c1 = f.read()
with open(f2, "r") as f: c2 = f.read()

# ==========================================
# 步骤 1: 撤销上次操作，恢复为原始 print 结构
# ==========================================
c1 = re.sub(
    r"if \(batch_idx == 0\) or \(\(batch_idx \+ 1\) % 10 == 0\):\n\s+import tqdm\n\s+tqdm\.tqdm\.write\([\s\S]*?f\"tot=\{total_loss\.item\(\):\.4f\}\"\n\s+\)",
    """if (batch_idx == 0) or ((batch_idx + 1) % 100 == 0):
                    print(
                        "Three-view losses: "
                        f"clean={loss_clean.item():.6f}, "
                        f"incomplete={loss_incomplete.item():.6f}, "
                        f"full={loss_full.item():.6f}, "
                        f"total={total_loss.item():.6f}"
                    )""",
    c1
)
c2 = c2.replace(
    'import tqdm; tqdm.tqdm.write(f"[{batch_idx + 1}/{cfg.max_steps}] Loss: {normalized_loss.item():.4f}")',
    'print("Loss: ", normalized_loss.item())'
)

with open(f1, "w") as f: f.write(c1)
with open(f2, "w") as f: f.write(c2)
print("✓ [1/3] 已成功撤销上次操作，恢复原始代码。")

# ==========================================
# 步骤 2: 保存当前的安全备份 (.bak)
# ==========================================
shutil.copy(f1, f1 + ".bak")
shutil.copy(f2, f2 + ".bak")
print("✓ [2/3] 已生成原始文件备份 (.bak)。")

# ==========================================
# 步骤 3: 实施正确的修改（剔除引发崩溃的局部 import）
# ==========================================
c1 = re.sub(
    r"if \(batch_idx == 0\) or \(\(batch_idx \+ 1\) % 100 == 0\):\n\s+print\([\s\S]*?f\"total=\{total_loss\.item\(\):\.6f\}\"\n\s+\)",
    """if (batch_idx == 0) or ((batch_idx + 1) % 10 == 0):
                    tqdm.tqdm.write(
                        f"[{batch_idx + 1}/{cfg.max_steps}] "
                        f"clean={loss_clean.item():.4f}, "
                        f"inc={loss_incomplete.item():.4f}, "
                        f"full={loss_full.item():.4f}, "
                        f"tot={total_loss.item():.4f}"
                    )""",
    c1
)
c2 = c2.replace(
    'print("Loss: ", normalized_loss.item())',
    'tqdm.tqdm.write(f"[{batch_idx + 1}/{cfg.max_steps}] Loss: {normalized_loss.item():.4f}")'
)

with open(f1, "w") as f: f.write(c1)
with open(f2, "w") as f: f.write(c2)
print("✓ [3/3] 正确的终端视图结构已应用。")
