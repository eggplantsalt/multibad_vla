file_path = "vla-scripts/train/finetune_with_task.py"

with open(file_path, "r") as f:
    code = f.read()

# 将执行合并的条件判断直接改为 False，彻底跳过基座模型加载与权重合并
old_code = "if cfg.use_lora and cfg.merge_lora_during_training:"
new_code = "if False:  # 物理阻断：全面禁止在训练中重载基座模型与合并权重"

code = code.replace(old_code, new_code)

with open(file_path, "w") as f:
    f.write(code)

print("✓ 硬盘 I/O 死锁与显存击穿的源头已被彻底封死。")
