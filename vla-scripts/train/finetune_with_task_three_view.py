"""
finetune_with_task_three_view.py

第二阶段三视图（clean/incomplete/full）微调脚本。
在复用原有 Stage II 训练流程的基础上，当启用 --enable_three_view 时，
对同一 batch 执行三路前向并加权汇总。
"""

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import draccus
import torch
import tqdm
from accelerate import PartialState
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import wandb

# 保持与原脚本同目录导入，避免额外打包改动。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import finetune_with_task as base

from prismatic.vla.research.compositional import (
    ActivationStats,
    CueManager,
    ThreeViewBatchTransform,
    ThreeViewLosses,
    ViewBuilder,
    ViewType,
    build_log_dict,
    compute_activation_score,
    compute_three_view_loss,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


@dataclass
class FinetuneThreeViewConfig(base.FinetuneConfig):
    # fmt: off
    enable_three_view: bool = False
    view_seed: int = 7
    view_loss_weights: str = "1.0,1.0,1.0"    # clean、incomplete、full 的权重
    three_view_mode: str = "mixed"            # clean | incomplete | full | mixed
    cue_window_size: int = 8
    cue_config_path: str = ""
    # fmt: on


def load_cue_config(path: str) -> Mapping[str, object]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"cue_config_path not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def parse_view_loss_weights(spec: str) -> Dict[ViewType, float]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("view_loss_weights must have three comma-separated values")
    values = [float(p) for p in parts]
    return {
        ViewType.CLEAN: values[0],
        ViewType.INCOMPLETE: values[1],
        ViewType.FULL: values[2],
    }


def apply_three_view_mode(weights: Dict[ViewType, float], mode: str) -> Dict[ViewType, float]:
    mode = mode.lower()
    if mode == "mixed":
        return weights
    if mode in ("clean", "incomplete", "full"):
        target = ViewType(mode)
        return {
            ViewType.CLEAN: 1.0 if target == ViewType.CLEAN else 0.0,
            ViewType.INCOMPLETE: 1.0 if target == ViewType.INCOMPLETE else 0.0,
            ViewType.FULL: 1.0 if target == ViewType.FULL else 0.0,
        }
    raise ValueError("three_view_mode must be clean|incomplete|full|mixed")


def apply_selected_view_bias(weights: Dict[ViewType, float], selected: ViewType) -> Dict[ViewType, float]:
    """Down-weight non-selected views to couple selection with training."""
    biased = {}
    for view_type, weight in weights.items():
        biased[view_type] = weight if view_type == selected else weight * 0.5
    return biased


def build_three_view_components(cfg: FinetuneThreeViewConfig) -> ThreeViewBatchTransform:
    cue_config = load_cue_config(cfg.cue_config_path)
    cue_manager = CueManager(cue_config=cue_config, seed=cfg.view_seed)
    view_builder = ViewBuilder(cue_manager)
    return ThreeViewBatchTransform(lambda sample: sample, view_builder)


@draccus.wrap()
def finetune(cfg: FinetuneThreeViewConfig) -> None:
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    run_id = base.get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb.init(mode="disabled")

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    if base.model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = base.snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
    else:
        AutoConfig.register("openvla", base.OpenVLAConfig)
        AutoImageProcessor.register(base.OpenVLAConfig, base.PrismaticImageProcessor)
        AutoProcessor.register(base.OpenVLAConfig, base.PrismaticProcessor)
        AutoModelForVision2Seq.register(base.OpenVLAConfig, base.OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        base.update_auto_map(cfg.vla_path)
        base.check_model_logic_mismatch(cfg.vla_path)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if cfg.use_lora:
        lora_model = base.find_target_modules(vla)
        lora_config = base.LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=lora_model,
            init_lora_weights="gaussian",
        )
        vla = base.get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
        vla.print_trainable_parameters()

    if cfg.use_film:
        base.count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = base.FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        base.count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = base.load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    vla = base.wrap_ddp(vla, device_id, find_unused=True)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = base.init_module(
            base.ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    action_head = None
    if cfg.use_l1_regression:
        action_head = base.init_module(
            base.L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    noisy_action_projector = None
    if cfg.use_diffusion:
        action_head = base.init_module(
            base.DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps": cfg.num_diffusion_steps,
            },
            to_bf16=True,
        )
        noisy_action_projector = base.init_module(
            base.NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    num_patches = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        num_patches += 1
    if cfg.use_diffusion:
        num_patches += 1

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=base.PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )

    three_view_transform = build_three_view_components(cfg)
    view_builder = three_view_transform.view_builder
    cue_manager = view_builder.cue_manager
    view_weights = parse_view_loss_weights(cfg.view_loss_weights)

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_clean": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_incomplete": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_full": deque(maxlen=cfg.grad_accumulation_steps),
        "total_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "activation_score_mean": deque(maxlen=cfg.grad_accumulation_steps),
    }

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            batch_size = batch["input_ids"].shape[0]

            if cfg.enable_three_view:
                cues = cue_manager.evaluate(batch)
                selected_view = cue_manager.choose_view_type(cues)
                activation_score = compute_activation_score(cues, cfg.cue_window_size)
                activation_stats = ActivationStats(score_mean=activation_score, window_size=cfg.cue_window_size)

                if cfg.three_view_mode == "mixed":
                    views = three_view_transform(batch)
                    loss_clean, _ = base.run_forward_pass(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        batch=views.clean,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        use_l1_regression=cfg.use_l1_regression,
                        use_diffusion=cfg.use_diffusion,
                        use_proprio=cfg.use_proprio,
                        use_film=cfg.use_film,
                        num_patches=num_patches,
                        compute_diffusion_l1=compute_diffusion_l1,
                        num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                    )
                    loss_incomplete, _ = base.run_forward_pass(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        batch=views.incomplete,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        use_l1_regression=cfg.use_l1_regression,
                        use_diffusion=cfg.use_diffusion,
                        use_proprio=cfg.use_proprio,
                        use_film=cfg.use_film,
                        num_patches=num_patches,
                        compute_diffusion_l1=compute_diffusion_l1,
                        num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                    )
                    loss_full, _ = base.run_forward_pass(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        batch=views.full,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        use_l1_regression=cfg.use_l1_regression,
                        use_diffusion=cfg.use_diffusion,
                        use_proprio=cfg.use_proprio,
                        use_film=cfg.use_film,
                        num_patches=num_patches,
                        compute_diffusion_l1=compute_diffusion_l1,
                        num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                    )
                    losses = ThreeViewLosses(
                        loss_clean=loss_clean,
                        loss_incomplete=loss_incomplete,
                        loss_full=loss_full,
                    )
                    step_weights = apply_selected_view_bias(view_weights, selected_view)
                    total_loss = compute_three_view_loss(losses, step_weights)
                    view_type_counts = {
                        "clean": batch_size,
                        "incomplete": batch_size,
                        "full": batch_size,
                    }
                    selected_view_sample = getattr(views, selected_view.value)
                else:
                    target_view = ViewType(cfg.three_view_mode)
                    target_batch = view_builder.apply_view(batch, target_view, cues=cues)
                    target_loss, _ = base.run_forward_pass(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        batch=target_batch,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        use_l1_regression=cfg.use_l1_regression,
                        use_diffusion=cfg.use_diffusion,
                        use_proprio=cfg.use_proprio,
                        use_film=cfg.use_film,
                        num_patches=num_patches,
                        compute_diffusion_l1=compute_diffusion_l1,
                        num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                    )
                    loss_clean = target_loss if target_view == ViewType.CLEAN else torch.tensor(0.0, device=device_id)
                    loss_incomplete = (
                        target_loss if target_view == ViewType.INCOMPLETE else torch.tensor(0.0, device=device_id)
                    )
                    loss_full = target_loss if target_view == ViewType.FULL else torch.tensor(0.0, device=device_id)
                    losses = ThreeViewLosses(
                        loss_clean=loss_clean,
                        loss_incomplete=loss_incomplete,
                        loss_full=loss_full,
                    )
                    total_loss = compute_three_view_loss(losses, apply_three_view_mode(view_weights, cfg.three_view_mode))
                    view_type_counts = {
                        "clean": batch_size if target_view == ViewType.CLEAN else 0,
                        "incomplete": batch_size if target_view == ViewType.INCOMPLETE else 0,
                        "full": batch_size if target_view == ViewType.FULL else 0,
                    }
                    selected_view_sample = target_batch

                if (batch_idx == 0) or ((batch_idx + 1) % 100 == 0):
                    print(
                        "Three-view losses: "
                        f"clean={loss_clean.item():.6f}, "
                        f"incomplete={loss_incomplete.item():.6f}, "
                        f"full={loss_full.item():.6f}, "
                        f"total={total_loss.item():.6f}"
                    )

                log_losses = ThreeViewLosses(
                    loss_clean=float(loss_clean.item()),
                    loss_incomplete=float(loss_incomplete.item()),
                    loss_full=float(loss_full.item()),
                )
                metrics = build_log_dict(
                    losses=log_losses,
                    activation=activation_stats,
                    view_type_counts=view_type_counts,
                )
                metrics["total_loss"] = float(total_loss.item())
                metrics["view_type_counts"] = view_type_counts
                metrics["selected_view_type"] = selected_view.value
                view_metadata = selected_view_sample.get("view_metadata", {})
                applied_cues = view_metadata.get("applied_cues", [])
                metrics["applied_cue_names"] = ",".join(applied_cues)
            else:
                total_loss, metrics = base.run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_diffusion=cfg.use_diffusion,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=num_patches,
                    compute_diffusion_l1=compute_diffusion_l1,
                    num_diffusion_steps=cfg.num_diffusion_steps if cfg.use_diffusion else None,
                )
                metrics["total_loss"] = float(total_loss.item())
                metrics["activation_score_mean"] = 0.0
                metrics["view_type_counts"] = {"clean": batch_size, "incomplete": 0, "full": 0}
                metrics["selected_view_type"] = "clean"
                metrics["applied_cue_names"] = ""

            normalized_loss = total_loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_metrics = base.compute_smoothened_metrics(recent_metrics)

            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_payload = dict(smoothened_metrics)
                if cfg.enable_three_view:
                    view_counts = metrics.get("view_type_counts", {})
                    log_payload["view_type_counts"] = (
                        f"clean={view_counts.get('clean', 0)},"
                        f"incomplete={view_counts.get('incomplete', 0)},"
                        f"full={view_counts.get('full', 0)}"
                    )
                    log_payload["activation_score_mean"] = metrics.get("activation_score_mean", 0.0)
                    log_payload["selected_view_type"] = metrics.get("selected_view_type", "")
                    log_payload["applied_cue_names"] = metrics.get("applied_cue_names", "")
                base.log_metrics_to_wandb(log_payload, "VLA Train", log_step, wandb)

            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                wandb.log({"VLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                base.save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                base.run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=num_patches,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                vla.train()

            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
