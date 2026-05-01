import re
import shutil

files = [
    "vla-scripts/train/finetune_with_task_three_view.py",
    "vla-scripts/train/finetune_with_task.py"
]

for filepath in files:
    try:
        with open(filepath, "r") as f:
            code = f.read()
            
        # 跳过已注入过的文件
        if "GradScaler" in code:
            print(f"[-] {filepath} 已包含 Scaler，跳过。")
            continue

        # 备份文件
        shutil.copy(filepath, filepath + ".amp_bak")

        # 1. 在训练循环启动前注入 Scaler 初始化 (替换首次出现的 optimizer.zero_grad)
        code = code.replace(
            "optimizer.zero_grad()", 
            "scaler = torch.cuda.amp.GradScaler()\n        optimizer.zero_grad()", 
            1
        )

        # 2. 拦截并缩放 Backward 信号
        code = code.replace(
            "normalized_loss.backward()", 
            "scaler.scale(normalized_loss).backward()"
        )

        # 3. 拦截 Optimizer 步进，注入解缩放、梯度裁剪与 Scaler 更新
        code = re.sub(
            r"(\s+)optimizer\.step\(\)\n\s+scheduler\.step\(\)",
            r"\1scaler.unscale_(optimizer)\1torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)\1scaler.step(optimizer)\1scaler.update()\1scheduler.step()",
            code
        )

        with open(filepath, "w") as f:
            f.write(code)
            
        print(f"[+] 成功在 {filepath} 中注入 AMP 梯度防线。")
        
    except FileNotFoundError:
        print(f"[!] 未找到文件 {filepath}")

