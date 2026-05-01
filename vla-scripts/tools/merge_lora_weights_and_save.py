"""
将仅包含 LoRA 适配器的 checkpoint 合并回基础 OpenVLA 模型，并保存合并后的权重。

请确保 base checkpoint 选择正确：
- 若微调的是默认 OpenVLA-7B 且未改模型代码，使用 `--base_checkpoint="openvla/openvla-7b"`。
- 若微调了其他模型或从其他 checkpoint 继续训练，请填写对应 base checkpoint。
- 若修改过 `modeling_prismatic.py`（OpenVLA 类实现），base checkpoint 需指向包含该修改的版本。

用法：
    python vla-scripts/tools/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class ConvertConfig:
    # fmt: off
    base_checkpoint: Union[str, Path] = ""                   # 基础模型 checkpoint 路径/目录
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""     # LoRA 适配器所在的 checkpoint 目录

    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    """合并 LoRA 权重并保存到原目录。"""
    # 注册 OpenVLA 到 HF Auto Classes（本地 checkpoint 必需）
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # 使用 HF AutoClasses 加载基础模型
    print(f"Loading base model: {cfg.base_checkpoint}")
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.base_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 加载 LoRA 权重，合并到基础模型并保存
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = PeftModel.from_pretrained(vla, os.path.join(cfg.lora_finetuned_checkpoint_dir, "lora_adapter")).to(
        "cuda"
    )
    merged_vla = merged_vla.merge_and_unload()
    merged_vla.save_pretrained(cfg.lora_finetuned_checkpoint_dir)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"\nSaved merged model checkpoint at:\n{cfg.lora_finetuned_checkpoint_dir}")


if __name__ == "__main__":
    main()
