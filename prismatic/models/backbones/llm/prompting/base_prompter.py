"""
base_prompter.py

Abstract class definition of a multi-turn prompt builder for ensuring consistent formatting for chat-based LLMs.
"""

from abc import ABC, abstractmethod
from typing import Optional


class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family

        # Only some models define a system prompt => let subclasses handle this logic!
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None: ...

    @abstractmethod
    def get_prompt(self) -> str: ...


class PurePromptBuilder(PromptBuilder):
    def __init__(
        self,
        model_family: str,
        system_prompt: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
    ) -> None:
        super().__init__(model_family, system_prompt)

        # 默认仅保证对 LLaMA 系列 token 兼容；其他模型可显式传入 bos/eos
        default_tokens = {
            "openvla": ("<s>", "</s>"),
            "llama": ("<s>", "</s>"),
            "llama2": ("<s>", "</s>"),
            "vicuna": ("<s>", "</s>"),
            "mistral": ("<s>", "</s>"),
            "phi": ("<|endoftext|>", "<|endoftext|>"),
            "phi-2": ("<|endoftext|>", "<|endoftext|>"),
        }
        if bos_token is None or eos_token is None:
            bos_default, eos_default = default_tokens.get(model_family, ("<s>", "</s>"))
            bos_token = bos_default if bos_token is None else bos_token
            eos_token = eos_default if eos_token is None else eos_token
        self.bos, self.eos = bos_token, eos_token

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()
