"""
action_tokenizer.py

动作离散化与编码器封装，用于将连续动作映射为 token。
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        将连续动作离散化为若干 bin，并映射到词表尾部的 token。

        注意：默认假设使用类似 LlamaTokenizer 的 BPE 分词器，词表尾部是较少使用的 token。

        :param tokenizer: 基础 LLM/VLM 分词器
        :param bins: 每个动作维度的离散 bin 数量（均匀划分）
        :param min_action: 动作最小值（用于裁剪）
        :param max_action: 动作最大值（用于裁剪）
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # 构建均匀分箱并计算 bin 中心
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # 约定：将动作 token 放在词表末尾 n_bins 个位置
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """将动作裁剪并映射到词表末尾的 n_bins 个 token。"""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        将离散动作 token 反解为连续动作值。

        说明：digitize 返回区间索引为 [1, bins]，但实际区间数为 bins-1，因此需要做裁剪。
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
