import re

file_path = "vla-scripts/train/finetune_with_task.py"

with open(file_path, "r") as f:
    code = f.read()

# 定位导致 OOM 的 PeftModel.from_pretrained 代码行
# 在其前方直接注入 return 语句，让保存函数在成功保存 LoRA 后立即安全退出
pattern = r"(\s+)merged_vla = PeftModel\.from_pretrained\("
replacement = r"\1# 物理阻断：禁止在训练中途重载基础模型合并权重引发显存雪崩\1return\n\1merged_vla = PeftModel.from_pretrained("

code = re.sub(pattern, replacement, code)

with open(file_path, "w") as f:
    f.write(code)

print("✓ 致命的在线权重合并逻辑已成功切断。")
