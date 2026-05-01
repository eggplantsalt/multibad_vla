"""机器人策略评测通用工具函数。"""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)

# 关键常量
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# NumPy 输出格式
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# OpenVLA v0.1 系统提示词
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# 模型图像尺寸配置
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}


def set_seed_everywhere(seed: int) -> None:
    """设置随机种子以确保可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg: Any, wrap_diffusion_policy_for_droid: bool = False) -> torch.nn.Module:
    """根据配置加载评测模型。"""
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg: Any) -> Union[int, tuple]:
    """获取指定模型的输入图像尺寸。"""
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return MODEL_IMAGE_SIZES[cfg.model_family]


def get_action(
    cfg: Any,
    model: torch.nn.Module,
    obs: Dict[str, Any],
    task_label: str,
    processor: Optional[Any] = None,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> Union[List[np.ndarray], np.ndarray]:
    """调用模型得到动作预测。"""
    with torch.no_grad():
        if cfg.model_family == "openvla":
            action = get_vla_action(
                cfg=cfg,
                vla=model,
                processor=processor,
                obs=obs,
                task_label=task_label,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=use_film,
            )
        else:
            raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """将夹爪动作从 [0,1] 归一化到 [-1,+1]。"""
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """反转夹爪动作符号（最后一维）。"""
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action
