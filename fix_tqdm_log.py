import re

# 1. 修复三视图扩展脚本中的日志打印
file_path_1 = "vla-scripts/train/finetune_with_task_three_view.py"
with open(file_path_1, "r") as f:
    code_1 = f.read()

# 匹配 445-452 行的原始 print 块
old_pattern = re.compile(
    r"if \(batch_idx == 0\) or \(\(batch_idx \+ 1\) % 100 == 0\):\n\s+print\([\s\S]*?f\"total=\{total_loss\.item\(\):\.6f\}\"\n\s+\)",
    re.MULTILINE
)

# 替换为 tqdm.write 结构，并降低输出间隔至 10 步
new_block_1 = """if (batch_idx == 0) or ((batch_idx + 1) % 10 == 0):
                    import tqdm
                    tqdm.tqdm.write(
                        f"[{batch_idx + 1}/{cfg.max_steps}] "
                        f"clean={loss_clean.item():.4f}, "
                        f"inc={loss_incomplete.item():.4f}, "
                        f"full={loss_full.item():.4f}, "
                        f"tot={total_loss.item():.4f}"
                    )"""

code_1 = old_pattern.sub(new_block_1, code_1)

with open(file_path_1, "w") as f:
    f.write(code_1)

# 2. 同步修复基础微调脚本中可能引发跳变的残余 print 语句
file_path_2 = "vla-scripts/train/finetune_with_task.py"
with open(file_path_2, "r") as f:
    code_2 = f.read()

# 替换基础脚本中的 print("Loss: ", normalized_loss.item())
code_2 = re.sub(
    r'print\("Loss: ", normalized_loss\.item\(\)\)', 
    r'import tqdm; tqdm.tqdm.write(f"[{batch_idx + 1}/{cfg.max_steps}] Loss: {normalized_loss.item():.4f}")', 
    code_2
)

with open(file_path_2, "w") as f:
    f.write(code_2)

print("IO重构完成。")
