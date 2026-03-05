#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distillation training codebase for Flux transformer models.

Trains a pruned (student) Flux transformer via knowledge distillation from
the original (teacher) model. Handles Flux-specific latent packing/unpacking.
Supports three loss components:
- L_task: flow matching diffusion loss against the ground-truth target.
- L_kd_out: output-level KD loss (student vs. teacher denoising predictions).
- L_kd_feat: feature-level KD loss (intermediate activations via forward hooks).
"""

import copy
import logging
import math
import os
import shutil
import time
from pathlib import Path

import diffusers
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from transformers import T5EncoderModel

from data.utils import collate_fn_img_txt
from data.ye_pop import YePopDataset
from model.builder import (
    get_DiffusersAPIBitsAndBytesConfig,
    get_TransformersAPIBitsAndBytesConfig,
    get_vae,
    load_and_cut_transformer,
    vae_decode,
    vae_encode,
)
from model.hook import add_hook
from pipelines.pipeline_dcae_flux import DCAEFluxPipeline
from utils.args import parse_args
from utils.identity_block import get_all_identity_blocks
from utils.utils import (
    adjust_SD3Transformer2DModel_inout4dcae,
    get_num_params,
    get_num_params_trainable,
    get_prompts,
    log_validation,
    parse_cut_blocks,
    return_none,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    validation_prompts=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {
                    "text": validation_prompts[i] if validation_prompts else " ",
                    "output": {"url": f"image_{i}.png"},
                }
            )

    model_description = f"""
# SD3 Distilled Model - {repo_id}

<Gallery />

## Model description

These are {repo_id} distilled weights for {base_model}.

The weights were trained using [ye-pop](https://huggingface.co/datasets/Ejafa/ye-pop).

## Download model

[Download]({repo_id}/tree/main) them in the Files & versions tab.

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        prompt=validation_prompts[0],
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "flux",
        "flux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        # CLIP text encoder's logging message regarding max prompt length is at warning level,
        # Hence, we change the logging level from info to error.
        diffusers.utils.logging.set_verbosity_error()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Load VAE
    vae = get_vae(
        name=args.vae_name,
        model_path=args.vae_pretrained,
        dtype=weight_dtype,
        device=accelerator.device,
        cache_dir=args.cache_dir,
    )
    vae.requires_grad_(False)

    ##### DISTILL CODE #####
    transformer_precision = vars(args).get("transformer_precision", None)
    quantization_config = get_DiffusersAPIBitsAndBytesConfig(
        precision=transformer_precision, weight_dtype=weight_dtype
    )

    transformer_teacher = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
        torch_dtype=weight_dtype,
        quantization_config=quantization_config,
    )
    transformer_teacher.requires_grad_(False)

    ##### Block-Removal or Fine-Grained Component Removal of Transformer Model #####
    (
        transformer,
        cut_blocks_l,
        cut_fg_dict,
        n_params_tx_orig,
        n_params_tx_cut,
    ) = load_and_cut_transformer(
        pretrained_name=args.pretrained_model_name_or_path,
        cut_transformer_blocks=args.cut_transformer_blocks,
        cut_transformer_blocks_2=args.cut_transformer_blocks_2,
        cut_transformer_components_excluded=args.cut_transformer_components_excluded,
        cut_transformer_type=args.cut_tx_type,
        target_memory_budget=float(args.target_memory_budget)
        if args.target_memory_budget is not None
        else None,
        benchmark_type=args.benchmark_type,
        metric_output_dir=args.metric_output_dir,
        cache_dir=args.cache_dir,
        debug=True,
    )

    ##### Knowledge Distillation Loss Scaling #####
    loss_scaling_range_l = None
    transformer_config_num_layers = transformer.config.num_layers
    if vars(args).get("kd_loss_scaling", None):
        identity_block_classes = get_all_identity_blocks()
        print(f"[INFO][Transformer] Enabling KD_LOSS_SCALING")
        transformer.requires_grad_(False)
        grad = False
        if vars(args).get("kd_loss_scaling_range", None):
            loss_scaling_range_l = parse_cut_blocks(args.kd_loss_scaling_range)
            for i, (n, m) in enumerate(transformer.named_modules()):
                if n.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                    idx = int(n.split(".")[1])
                    if n.startswith("single_transformer_blocks."):
                        idx += transformer_config_num_layers

                    if idx in loss_scaling_range_l:
                        grad = True
                    else:
                        grad = False
                m.requires_grad_(grad)
        else:
            for i, (n, m) in enumerate(transformer.named_modules()):
                if (
                    n.startswith(("transformer_blocks.", "single_transformer_blocks."))
                    and grad == True
                ):
                    idx = int(n.split(".")[1])
                    if n.startswith("single_transformer_blocks."):
                        idx += transformer_config_num_layers
                    if loss_scaling_range_l is None:
                        loss_scaling_range_l = [idx]
                    else:
                        loss_scaling_range_l.append(idx)

                m.requires_grad_(grad)
                if isinstance(m, identity_block_classes):
                    grad = True
    if loss_scaling_range_l is not None:
        # to remove the duplicate numbers in the list, we convert it to set
        loss_scaling_range_l = set(loss_scaling_range_l)

    if args.distill_type == "pixel":
        # TODO: codes below are relevant to SD3 transformers,
        # they may need to be updated to Flux Transformers
        adjust_SD3Transformer2DModel_inout4dcae(
            transformer,
            resolution=args.resolution,
            ae=vae,
            patch_size=args.transformer_patch_size,
            init_type=args.transformer_init_type,
        )

        # load vae_teacher
        # TODO: need to update the hard-coded part of giving pretrained model
        vae_teacher = get_vae(
            "flux", "stabilityai/stable-diffusion-3.5-medium", accelerator.device, args.cache_dir
        )
        vae_teacher.requires_grad_(False)

    print(
        f"[INFO][Transformer] Removed {cut_blocks_l} transformer blocks from the original transformer"
    )
    print(
        f"[INFO][Transformer] {n_params_tx_cut/1e9:0.2f}B params remain out of {n_params_tx_orig/1e9:0.2f}B total params ({n_params_tx_cut/n_params_tx_orig*100:0.2f}%)"
    )
    print(
        f"[INFO][Transformer] {get_num_params_trainable(transformer)/1e9:0.2f}B trainable params out of {get_num_params(transformer)/1e9:0.2f}B total params ({get_num_params_trainable(transformer)/get_num_params(transformer)*100:0.2f}%)"
    )
    print(
        f"[INFO][Transformer] {get_num_params_trainable(transformer)/1e9:0.2f}B trainable params out of ORIGINAL {get_num_params(transformer_teacher)/1e9:0.2f}B total params ({get_num_params_trainable(transformer)/get_num_params(transformer_teacher)*100:0.2f}%)"
    )

    # Text Encoder Pipeline
    t5_precision = vars(args).get("t5_precision", None)
    quantization_config = get_TransformersAPIBitsAndBytesConfig(
        precision=t5_precision, weight_dtype=weight_dtype
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        cache_dir=args.cache_dir,
        torch_dtype=weight_dtype,
        quantization_config=quantization_config,
    )
    text_encoder.requires_grad_(False)
    pipeline_text_data = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder_2=text_encoder,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype,
        cache_dir=args.cache_dir,
    )

    if args.torch_compile:
        compile_time_start = time.time()
        vae = torch.compile(vae)
        pipeline_text_data.text_encoder = torch.compile(pipeline_text_data.text_encoder)
        pipeline_text_data.text_encoder_2 = torch.compile(pipeline_text_data.text_encoder_2)
        # Applying torch.compile on a trainable model seems to make gradient_accumulation not working.
        # Uncomment the line below causes OOM.
        # transformer = torch.compile(transformer)
        transformer_teacher = torch.compile(transformer_teacher)
        compile_time = time.time() - compile_time_start
        print(f"[INFO] Compile Time {compile_time:0.2f}")

    # Q: Why VAE dtype is casted to torch.float32?
    vae.to(accelerator.device, dtype=weight_dtype)
    pipeline_text_data.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer_teacher.to(accelerator.device, dtype=weight_dtype)
    if args.distill_type == "pixel":
        if args.use_torch_compile:
            vae_teacher = torch.compile(vae_teacher)
        vae_teacher.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_parameters_with_lr = {
        "params": transformer_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not args.optimizer.lower() == "adamw":
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include [adamW]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    ##### CUSTOM DATASET #####
    train_dataset = YePopDataset(
        path=args.instance_data_dir,
        random_flip=args.random_flip,
        size=args.resolution,
        caption_type=args.caption_type,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_img_txt,
        drop_last=True,
        prefetch_factor=2,
        pin_memory=True,
    )

    validation_prompts = get_prompts(args.validation_prompts_config)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    try:
        num_update_steps_per_epoch = math.ceil(
            len(train_dataset) / args.gradient_accumulation_steps
        )
    except:
        num_update_steps_per_epoch = math.ceil(
            (500 * 1000 / args.train_batch_size) / args.gradient_accumulation_steps
        )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    try:
        num_examples = len(train_dataset)
        num_batches_per_epoch = math.ceil(len(train_dataset) / args.train_batch_size)
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
    except:
        num_examples = 500 * 1000
        num_batches_per_epoch = math.ceil(500 * 1000 / args.train_batch_size)
        num_update_steps_per_epoch = math.ceil(
            (500 * 1000 / args.train_batch_size / accelerator.num_processes)
            / args.gradient_accumulation_steps
        )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux-distilled"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Num batches each epoch = {num_batches_per_epoch}")
    logger.info(f"  Num iterations each epoch = {len(train_dataloader)}")
    logger.info(
        f"  Num effective update steps (iterations/grad accumulation steps) each epoch = {num_update_steps_per_epoch}"
    )
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Validation Prompts = {validation_prompts}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step_in_epoch = global_step % num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )

    ### Add hooks for feature distillation ###
    if args.lambda_kd_feat > 0.0:

        mapping_layers = []

        if is_compiled_module(transformer_teacher):
            filter_str = ("_orig_mod.transformer_blocks.", "_orig_mod.single_transformer_blocks.")
            module_name_length = 3
        else:
            filter_str = ("transformer_blocks.", "single_transformer_blocks.")
            module_name_length = 2

        if "SA" in args.kd_feat_type or "CA" in args.kd_feat_type:  # attn is included
            num_added_layers = 0
            for i, (n, m) in enumerate(transformer_teacher.named_modules()):
                if n.endswith(".attn"):
                    block_idx = int(n.split(".")[module_name_length - 1])
                    if "single_transformer_blocks." in n:
                        block_idx += transformer_config_num_layers
                    if block_idx not in cut_blocks_l:
                        if (
                            loss_scaling_range_l is None
                        ):  # when loss_scaling_range_l is not specified, we simply add all layers for feature distillation
                            mapping_layers.append(n)
                            num_added_layers += 1
                        elif loss_scaling_range_l is not None and block_idx in loss_scaling_range_l:
                            mapping_layers.append(n)
                            num_added_layers += 1
            print(
                f"[INFO][Transformer][KD] num_attn (SA,CA): {num_added_layers}, total_num_mapped_layers: {len(mapping_layers)}"
            )

        if "SA2" in args.kd_feat_type:  # attn2 is included
            num_added_layers = 0
            for i, (n, m) in enumerate(transformer_teacher.named_modules()):
                if n.endswith(".attn2"):
                    block_idx = int(n.split(".")[module_name_length - 1])
                    if "single_transformer_blocks." in n:
                        block_idx += transformer_config_num_layers
                    if block_idx not in cut_blocks_l:
                        if (
                            loss_scaling_range_l is None
                        ):  # when loss_scaling_range_l is not specified, we simply add all layers for feature distillation
                            mapping_layers.append(n)
                            num_added_layers += 1
                        elif loss_scaling_range_l is not None and block_idx in loss_scaling_range_l:
                            mapping_layers.append(n)
                            num_added_layers += 1
            print(
                f"[INFO][Transformer][KD] num_attn2 (SA2): {num_added_layers}, total_num_mapped_layers: {len(mapping_layers)}"
            )

        if "LFImg" in args.kd_feat_type or "LFCond" in args.kd_feat_type:  # LastFeature is included
            num_added_layers = 0
            for i, (n, m) in enumerate(transformer_teacher.named_modules()):
                if n.startswith(filter_str) and len(n.split(".")) == module_name_length:
                    block_idx = int(n.split(".")[module_name_length - 1])
                    if "single_transformer_blocks." in n:
                        block_idx += transformer_config_num_layers
                    if block_idx not in cut_blocks_l:
                        if (
                            loss_scaling_range_l is None
                        ):  # when loss_scaling_range_l is not specified, we simply add all layers for feature distillation
                            mapping_layers.append(n)
                            num_added_layers += 1
                        elif loss_scaling_range_l is not None and block_idx in loss_scaling_range_l:
                            mapping_layers.append(n)
                            num_added_layers += 1
            print(
                f"[INFO][Transformer][KD] num_LastFeat (LF): {num_added_layers}, total_num_mapped_layers: {len(mapping_layers)}"
            )

        mapping_layers_teacher = copy.deepcopy(mapping_layers)
        mapping_layers_student = copy.deepcopy(mapping_layers)

        if is_compiled_module(transformer_teacher):
            for i, m_stu in enumerate(mapping_layers_student):
                mapping_layers_student[i] = m_stu.replace("_orig_mod.", "")

        if torch.cuda.device_count() > 1:
            print(f"[INFO] Num GPUs: {torch.cuda.device_count()}")
            # revise the hooked feature names for student (to consider ddp wrapper)
            for i, m_stu in enumerate(mapping_layers_student):
                mapping_layers_student[i] = "module." + m_stu

        acts_teacher = {}
        acts_student = {}
        add_hook(transformer_teacher, acts_teacher, mapping_layers_teacher)
        add_hook(transformer, acts_student, mapping_layers_student)

    ### Start Training Loop ###
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        data_time_start = time.time()
        data_time_all = 0
        vae_time_all = 0
        lm_time_all = 0
        model_time_all = 0

        train_loss = 0.0
        train_loss_task = 0.0
        train_loss_kd_out = 0.0
        train_loss_kd_feat = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip until it reaches the resumed step in an epoch
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step_in_epoch:
                continue

            models_to_accumulate = [transformer]
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start

            # Convert images to latent space
            vae_time_start = time.time()
            model_input = vae_encode(
                args.vae_name,
                vae,
                batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype),
                sample_posterior=True,
            ).to(dtype=weight_dtype)
            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start

            # Convert text to embedding
            lm_time_start = time.time()
            if hasattr(batch, "prompt_embeds"):
                prompt_embeds, pooled_prompt_embeds = (
                    batch["prompt_embeds"],
                    batch["pooled_prompt_embeds"],
                )
                prompt_embeds = prompt_embeds.squeeze(1).to(
                    device=accelerator.device, dtype=weight_dtype
                )
                pooled_prompt_embeds = pooled_prompt_embeds.squeeze(1).to(
                    device=accelerator.device, dtype=weight_dtype
                )
                text_ids = torch.squeeze(batch["text_ids"], 0).to(device=accelerator.device)
            else:
                with torch.no_grad():
                    (
                        prompt_embeds,
                        pooled_prompt_embeds,
                        text_ids,
                    ) = pipeline_text_data.encode_prompt(
                        prompt=batch["prompt"],
                        prompt_2=None,
                        max_sequence_length=args.max_sequence_length,
                    )
                    prompt_embeds = prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(
                        device=accelerator.device, dtype=weight_dtype
                    )
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start

            # Start denoising process
            model_time_start = time.time()
            with accelerator.accumulate(models_to_accumulate):
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = texp * z1 + (1 - texp) * x
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

                # pack latents [B, C, H, W] -> [B, 4C, H/2, W/2]
                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                if args.lambda_task > 0.0:
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    # Preconditioning of the model outputs.
                    model_pred_sd_task = model_pred * (-sigmas) + noisy_model_input

                if args.distill_type == "pixel":
                    model_pred = vae_decode(args.vae_name, vae, model_pred.to(weight_dtype)).to(
                        dtype=weight_dtype
                    )

                with torch.no_grad():
                    if args.distill_type == "pixel":
                        model_input_teacher = vae_encode(
                            "flux",
                            vae_teacher,
                            batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype),
                            sample_posterior=True,
                        ).to(dtype=weight_dtype)
                        noise_teacher = torch.randn_like(model_input_teacher)
                        noise_teacher.flatten()[
                            : noise.shape[0] * noise.shape[1] * noise.shape[2] * noise.shape[3]
                        ] = noise.flatten()[:]
                        noisy_model_input_teacher = (
                            sigmas * noise_teacher + (1.0 - sigmas) * model_input_teacher
                        )
                        model_input_teacher_shape = model_input_teacher.shape
                        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                            model_input_teacher_shape[0],
                            model_input_teacher_shape[2] // 2,
                            model_input_teacher_shape[3] // 2,
                            accelerator.device,
                            weight_dtype,
                        )
                    else:
                        noisy_model_input_teacher = noisy_model_input
                        model_input_teacher_shape = model_input.shape

                    # pack latents [B, C, H, W] -> [B, 4C, H/2, W/2]
                    packed_noisy_model_input_teacher = FluxPipeline._pack_latents(
                        noisy_model_input_teacher,
                        batch_size=model_input_teacher_shape[0],
                        num_channels_latents=model_input_teacher_shape[1],
                        height=model_input_teacher_shape[2],
                        width=model_input_teacher_shape[3],
                    )
                    model_pred_teacher = transformer_teacher(
                        hidden_states=packed_noisy_model_input_teacher,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                    model_pred_teacher = FluxPipeline._unpack_latents(
                        model_pred_teacher,
                        height=model_input_teacher_shape[2] * vae_scale_factor,
                        width=model_input_teacher_shape[3] * vae_scale_factor,
                        vae_scale_factor=vae_scale_factor,
                    )

                    if args.distill_type == "pixel":
                        model_pred_teacher = vae_decode(
                            "flux", vae_teacher, model_pred_teacher.to(weight_dtype)
                        ).to(dtype=weight_dtype)

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                ### Compute Losses ((1) L_task, (2) L_kd_out, (3) L_kd_feat)
                # Compute diffusion Task Loss (L_task)
                if args.lambda_task > 0.0:
                    # Get sd task's target
                    # flow matching loss
                    target = model_input

                    loss_task = torch.mean(
                        (
                            weighting.float() * (model_pred_sd_task.float() - target.float()) ** 2
                        ).reshape(target.shape[0], -1),
                        1,
                    ).mean()
                else:
                    loss_task = torch.tensor(0.0)

                # Compute KD-Output Loss (L_kd_out)
                if args.lambda_kd_out > 0.0:
                    loss_kd_out = torch.mean(
                        (
                            weighting.float()
                            * (model_pred.float() - model_pred_teacher.float()) ** 2
                        ).reshape(model_pred_teacher.shape[0], -1),
                        1,
                    ).mean()
                else:
                    loss_kd_out = torch.tensor(0.0)

                # Compute KD-feat Loss (L_kd_feat)
                if args.lambda_kd_feat > 0.0:
                    losses_kd_feat = []
                    for m_teacher, m_student in zip(mapping_layers_teacher, mapping_layers_student):
                        a_teacher = acts_teacher[m_teacher]
                        a_student = acts_student[m_student]

                        # attn_output, context_attn_output = self.attn(...) attn_output==SA / context_attn_output==CA
                        # norm_hidden_states = self.norm2(hidden_states)
                        # encoder_hidden_states, hidden_states = tx_block(...) encoder==LFCond / hidden_states==LFImg
                        # hidden_states = single_tx_block(...) encoder==LFCond / hidden_states==LFImg
                        if type(a_teacher) is tuple and type(a_student) is tuple:
                            if not (
                                "LFCond" in args.kd_feat_type
                                or "SA" in args.kd_feat_type
                                or "LFImg" in args.kd_feat_type
                                or "CA" in args.kd_feat_type
                            ):
                                raise ValueError()

                            if m_teacher.endswith(".attn"):
                                if "SA" in args.kd_feat_type and (
                                    a_student[0] is not None and a_teacher[0] is not None
                                ):
                                    tmp = torch.mean(
                                        (
                                            weighting.float()
                                            * (a_student[0].float() - a_teacher[0].detach().float())
                                            ** 2
                                        ).reshape(a_teacher[0].shape[0], -1),
                                        1,
                                    ).mean()
                                    losses_kd_feat.append(tmp)
                                if "CA" in args.kd_feat_type and (
                                    a_student[1] is not None and a_teacher[1] is not None
                                ):
                                    tmp = torch.mean(
                                        (
                                            weighting.float()
                                            * (a_student[1].float() - a_teacher[1].detach().float())
                                            ** 2
                                        ).reshape(a_teacher[1].shape[0], -1),
                                        1,
                                    ).mean()
                                    losses_kd_feat.append(tmp)
                            elif len(m_teacher.split(".")) <= module_name_length:
                                if "LFCond" in args.kd_feat_type and (
                                    a_student[0] is not None and a_teacher[0] is not None
                                ):
                                    tmp = torch.mean(
                                        (
                                            weighting.float()
                                            * (a_student[0].float() - a_teacher[0].detach().float())
                                            ** 2
                                        ).reshape(a_teacher[0].shape[0], -1),
                                        1,
                                    ).mean()
                                    losses_kd_feat.append(tmp)
                                if "LFImg" in args.kd_feat_type and (
                                    a_student[1] is not None and a_teacher[1] is not None
                                ):
                                    tmp = torch.mean(
                                        (
                                            weighting.float()
                                            * (a_student[1].float() - a_teacher[1].detach().float())
                                            ** 2
                                        ).reshape(a_teacher[1].shape[0], -1),
                                        1,
                                    ).mean()
                                    losses_kd_feat.append(tmp)
                            else:
                                raise NotImplementedError()
                        else:
                            tmp = torch.mean(
                                (
                                    weighting.float()
                                    * (a_student.float() - a_teacher.detach().float()) ** 2
                                ).reshape(a_teacher.shape[0], -1),
                                1,
                            ).mean()
                            losses_kd_feat.append(tmp)
                    loss_kd_feat = sum(losses_kd_feat)
                else:
                    loss_kd_feat = torch.tensor(0.0)

                # Calculate the final loss
                loss = (
                    args.lambda_task * loss_task
                    + args.lambda_kd_out * loss_kd_out
                    + args.lambda_kd_feat * loss_kd_feat
                )

                # Gather the losses across all processes for logging in case of distributed training.
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                if args.lambda_task > 0.0:
                    avg_loss_task = accelerator.gather(
                        loss_task.repeat(args.train_batch_size)
                    ).mean()
                    train_loss_task += avg_loss_task.item() / args.gradient_accumulation_steps

                if args.lambda_kd_out > 0.0:
                    avg_loss_kd_out = accelerator.gather(
                        loss_kd_out.repeat(args.train_batch_size)
                    ).mean()
                    train_loss_kd_out += avg_loss_kd_out.item() / args.gradient_accumulation_steps

                if args.lambda_kd_feat > 0.0:
                    avg_loss_kd_feat = accelerator.gather(
                        loss_kd_feat.repeat(args.train_batch_size)
                    ).mean()
                    train_loss_kd_feat += avg_loss_kd_feat.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "train_loss_task": train_loss_task,
                        "train_loss_kd_out": train_loss_kd_out,
                        "train_loss_kd_feat": train_loss_kd_feat,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "data_time": data_time_all * args.gradient_accumulation_steps,
                        "vae_time": vae_time_all * args.gradient_accumulation_steps,
                        "lm_time": lm_time_all * args.gradient_accumulation_steps,
                        "model_time": model_time_all * args.gradient_accumulation_steps,
                    },
                    step=global_step,
                )

                train_loss = 0.0
                train_loss_task = 0.0
                train_loss_kd_out = 0.0
                train_loss_kd_feat = 0.0

                # Save checkpoints
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.state.deepspeed_plugin is None:
                        if accelerator.is_main_process:
                            accelerator.save_state(save_path)
                    else:
                        # DeepSpeed requires all processes to execute save_state function.
                        # Hence we do not wrap save_state function within accelerator.is_main_process when DeepSpeed is used
                        accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    # Perform validation and display intermediate results (images) on tensorboard and/or on output_dir/logging_dir
                    if validation_prompts is not None and global_step % args.validation_steps == 0:
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {validation_prompts}."
                        )

                        # Initialise pipeline and load other components
                        if args.vae_name == "dc-ae":
                            pipeline = DCAEFluxPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                vae=None,
                                transformer=accelerator.unwrap_model(
                                    transformer, keep_fp32_wrapper=True
                                ),
                                text_encoder_2=text_encoder,
                                torch_dtype=weight_dtype,
                                cache_dir=args.cache_dir,
                            )
                            pipeline.vae = vae
                            pipeline.vae_scale_factor = 2 ** sum(
                                tmp > 0 for tmp in vae.encoder.cfg.depth_list[:-1]
                            )
                        else:
                            pipeline = FluxPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                vae=vae,
                                transformer=accelerator.unwrap_model(
                                    transformer, keep_fp32_wrapper=True
                                ),
                                text_encoder_2=text_encoder,
                                torch_dtype=weight_dtype,
                                cache_dir=args.cache_dir,
                            )

                        # args as an input for pipeline
                        pipeline_args = {
                            "prompt": validation_prompts,
                            "negative_prompt": "",
                            "height": args.resolution,
                            "width": args.resolution,
                            "num_inference_steps": args.num_inference_steps,
                            "guidance_scale": args.guidance_scale,
                        }
                        images = log_validation(
                            pipeline=pipeline,
                            pipeline_args=pipeline_args,
                            global_step=global_step,
                            args=args,
                            device=accelerator.device.type,
                            trackers=accelerator.trackers,
                            is_final_validation=False,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "loss_task": loss_task.detach().item(),
                "loss_kd_out": loss_kd_out.detach().item(),
                "loss_kd_feat": loss_kd_feat.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "data_time": data_time_all,
                "vae_time": vae_time_all,
                "lm_time": lm_time_all,
                "model_time": model_time_all,
            }

            progress_bar.set_postfix(**logs)
            data_time_all = 0
            vae_time_all = 0
            lm_time_all = 0
            model_time_all = 0

            if global_step >= args.max_train_steps:
                break

            data_time_start = time.time()

    # Save the distilled model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Final inference
        # Load previous pipeline
        if args.vae_name == "dc-ae":
            pipeline = DCAEFluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=None,
                transformer=accelerator.unwrap_model(transformer),
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                cache_dir=args.cache_dir,
            )
            pipeline.vae = vae
            pipeline.vae_scale_factor = 2 ** sum(tmp > 0 for tmp in vae.encoder.cfg.depth_list[:-1])
        else:
            pipeline = FluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                transformer=accelerator.unwrap_model(transformer),
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                cache_dir=args.cache_dir,
            )
        pipeline.save_pretrained(args.output_dir)

        # run inference
        images = []
        if validation_prompts and args.num_validation_images > 0:
            pipeline_args = {
                "prompt": validation_prompts,
                "negative_prompt": "",
                "height": args.resolution,
                "width": args.resolution,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
            }
            images = log_validation(
                pipeline=pipeline,
                pipeline_args=pipeline_args,
                global_step=global_step,
                args=args,
                device=accelerator.device.type,
                trackers=accelerator.trackers,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                validation_prompts=validation_prompts,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
