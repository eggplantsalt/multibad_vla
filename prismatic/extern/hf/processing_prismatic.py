"""
processing_prismatic.py

Prismatic VLM 的 HuggingFace 预处理器定义，继承自 `ProcessorMixin`。
默认配置对应 `siglip-224px+7b`。
"""

from typing import Any, ClassVar, List, Optional, Tuple, Union

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


# === 图像预处理 ===
def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """将 PIL 图像等比例补边到正方形。"""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        """
        初始化 PrismaticImageProcessor：基于 TIMM 生成 torchvision transform，并按
        `image_resize_strategy` 进行策略化改造。
        @param use_fused_vision_backbone: 是否使用融合（双）视觉骨干
        @param image_resize_strategy: resize 策略 < resize-naive | resize-crop | letterbox >
        @param input_size: [TIMM :: data_cfg] 输入图像大小 (C, W, H)
        @param interpolation: [TIMM :: data_cfg] 插值方式（默认 bicubic）
        @param mean: [TIMM :: data_cfg] 归一化均值
        @param std: [TIMM :: data_cfg] 归一化方差
        """
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # 处理默认参数
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        # TIMM data_cfg 参数
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        # 从 TIMM 获取 torchvision transforms，并解析为 functional 参数
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,  # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)
                crop_mode="center",  # Default crop mode -- no-op when `crop_pct == 1.0`
                is_training=False,  # No image augmentations when loading the transform!
            )

            # 校验 transform 结构与尺寸
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")

            # HF Image Processor 必须可 JSON 序列化，因此不直接持有 torchvision 对象
            # 改为解析 transform 并调用 torchvision.transforms.functional
            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

            # 处理 Prismatic 的 image_resize_strategy
            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == "resize-crop":
                # 保持默认 resize + center-crop 逻辑
                pass
            else:
                raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")

        # 透传 **kwargs 给父类
        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        """按 TIMM 规范执行 Resize -> CenterCrop -> ToTensor -> Normalize。"""
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        # 融合骨干期望通道堆叠输入，模型侧再拆分
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)

        # imgs_t 为 [3, H, W] 列表，沿 dim=0 堆叠
        img_t = torch.vstack(imgs_t)

        return img_t

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        预处理单张或多张图像（仅支持 PIL.Image）。
        @param images: 单张或多张 PIL.Image
        @param return_tensors: 输出张量类型（如 "pt"）；为空时返回 np.ndarray
        @return: BatchFeature，包含 "pixel_values"
        """
        if not isinstance(images, list):
            images = [images]

        # 逐张应用变换并堆叠为 batch Tensor
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])

        # BatchFeature 需要 np.ndarray，因此先转为 numpy
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
