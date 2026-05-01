"""
deploy.py

启动 VLA 推理服务端，客户端通过 /act 接口获取动作。
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === 服务端接口 ===
class OpenVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # 加载模型
        self.vla = get_vla(cfg)

        # 加载本体感受投影层
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # 加载连续动作头
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # 检查动作反归一化键
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # 获取 HuggingFace 处理器
        self.processor = None
        self.processor = get_processor(cfg)

        # 获取图像输入尺寸
        self.resize_size = get_image_resize_size(cfg)


    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            instruction = observation["instruction"]

            action = get_vla_action(
                self.cfg, self.vla, self.processor, observation, instruction, action_head=self.action_head, proprio_projector=self.proprio_projector, use_film=self.cfg.use_film,
            )

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # 服务端配置
    host: str = "0.0.0.0"                                               # 监听 IP
    port: int = 8777                                                    # 监听端口

    #################################################################################################################
    # 模型相关参数
    #################################################################################################################
    model_family: str = "openvla"                    # 模型家族
    pretrained_checkpoint: Union[str, Path] = ""     # 预训练/微调 checkpoint 路径

    use_l1_regression: bool = True                   # 是否使用 L1 回归动作头
    use_diffusion: bool = False                      # 是否使用扩散式动作头（DDIM）
    num_diffusion_steps_train: int = 50              # 扩散训练步数（use_diffusion=True 时）
    num_diffusion_steps_inference: int = 50          # 扩散推理步数（use_diffusion=True 时）
    use_film: bool = False                           # 是否启用 FiLM 视觉-语言调制
    num_images_in_input: int = 3                     # 输入图像数量（默认 3）
    use_proprio: bool = True                         # 是否使用本体感受输入

    center_crop: bool = True                         # 是否中心裁剪（训练时有随机裁剪则应为 True）

    lora_rank: int = 32                              # LoRA rank（需与训练一致）

    unnorm_key: Union[str, Path] = ""                # 动作反归一化键
    use_relative_actions: bool = False               # 是否使用相对动作（关节角增量）

    load_in_8bit: bool = False                       # 是否使用 8bit 量化加载（仅 OpenVLA）
    load_in_4bit: bool = False                       # 是否使用 4bit 量化加载（仅 OpenVLA）

    #################################################################################################################
    # 其他
    #################################################################################################################
    seed: int = 7                                    # 随机种子（可复现）
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
