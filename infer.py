"""Inference code for original/pruned/distilled SD3 and Flux diffusion models.

Generates images from text prompts using an original model or a pruned/distilled model.
Supports both quantised and non-quantised models.
"""

import argparse
import datetime
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from model.builder import load_and_cut_transformer_return_pipeline, load_distilled_pipeline
from utils.utils import get_prompts, return_none


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--transformer_pretrained",
        type=return_none,
        default=None,
        required=False,
        help="Path to pretrained Transformer model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_quant_type",
        type=return_none,
        default=None,
        required=False,
        help="Quantisation type for transformer. ['bnb_nf4', etc.]",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--benchmark_type",
        type=str,
        default="hpsv2",
        help="Data type for performing evaluation.",
    )
    parser.add_argument(
        "--benchmark_take",
        type=return_none,
        default=None,
        help="The number of images that are taken for evaluation. Default is None, which means all samples are used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--metric_output_dir",
        type=return_none,
        default=None,
        help="The output directory where the pre-computed metric results are stored.",
    )
    parser.add_argument(
        "--cache_dir",
        type=return_none,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--use_rand_seed",
        action="store_true",
        help="Use random seed for inference pipeline which override the seed argument above.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=("The height for input images, if height and width are given, max_res is ignored."),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=("The width for input images, if height and width are given, max_res is ignored."),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help=("The number of inference steps that the pipeline takes as an input."),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help=(
            "The guidance scale for balancing the strength of the text prompt as well as image."
            "The higher the guidance scale is, the more text prompt is considered when generating an image."
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help=("The maximum sequence length that the pipeline takes as an input for the prompt."),
    )
    parser.add_argument(
        "--validation_prompt",
        type=return_none,
        default=None,
        help="A prompt given to a model.",
    )
    parser.add_argument(
        "--validation_prompts_config",
        type=str,
        default="configs/val_prompts.yml",
        help="A list of prompts config path.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--cut_transformer_blocks",
        type=return_none,
        default=None,
        help=(
            "For removing transformer blocks. usage example:"
            "(1) when specifying individual block index, use 10,20"
            "(2) when giving a range of indices, use 10-20"
            "(3) when giving a list of ranges of indices, use 8-12,16-20"
        ),
    )
    parser.add_argument(
        "--cut_transformer_blocks_2",
        type=return_none,
        default=None,
        help=(
            "For removing transformer blocks' 2nd Range for hybrid distillation type."
            "Same usage example as above cut_transformer_blocks"
        ),
    )
    parser.add_argument(
        "--cut_transformer_components_excluded",
        type=return_none,
        default=None,
        help=(
            "When removing transformer sub-components, this is a exclusion list."
            "Example: attn or attn,norm1 or attn,norm1,norm1_context, attn,ff etc."
        ),
    )
    parser.add_argument(
        "--cut_tx_type",
        default="none",
        choices=[
            "none",
            "cut_blk_manual",
            "cut_blk_least_drop",
            "cut_fg_least_drop",
            "cut_fg_least_drop_param",
            "cut_hybrid",
        ],
        help=(
            "The type of pruning transformer."
            "(1) cut_blk_manual: prune the entire transformer block given cut_transformer_blocks: (default)"
            "(2) cut_blk_least_drop: prune the entire transformer block based on pre-computed metrics, choose the least important blocks for removal"
            "(3) cut_fg_least_drop: prune the fine-grained component in a transformer block based on pre-computed metrics, choose the least important components for removal"
            "(4) cut_fg_least_drop_param: prune the fine-grained component in a transformer block based on pre-computed metrics, choose the least important components per parameter for removal"
        ),
    )
    parser.add_argument(
        "--target_memory_budget",
        type=return_none,
        default=None,
        help=(
            "Memory budget for the remaining transformer model. usage example:"
            "If you want to remain 80'%' of the transformer model, give 0.8 here."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print out information that could be useful for debugging.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    # Seeding for reproducibility
    os.environ["PL_GLOBAL_SEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(True)}"

    # Mixed precision inference
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Image width and height arguments
    if args.height is None:
        height = args.resolution
    else:
        height = args.height
    if args.width is None:
        width = args.resolution
    else:
        width = args.width

    # Load Pipeline
    # use case 1: loading original/quantised pipeline (no cutting, distillation applied)
    # use case 2: loading distilled transformer from a checkpoint file.
    if (args.transformer_pretrained is None and args.cut_tx_type == "none") or (
        args.transformer_pretrained is not None and "checkpoint-" in args.transformer_pretrained
    ):
        pipe, cut_blocks_l, n_params_tx_orig, n_params_tx_cut = load_distilled_pipeline(
            pretrained_name=args.pretrained_model_name_or_path,
            transformer_pretrained=args.transformer_pretrained,
            transformer_quant_type=args.transformer_quant_type,
            cut_transformer_blocks=args.cut_transformer_blocks,
            weight_dtype=weight_dtype,
            cache_dir=args.cache_dir,
            debug=args.debug,
        )
        cut_fg_dict = None
    else:
        (
            pipe,
            cut_blocks_l,
            cut_fg_dict,
            n_params_tx_orig,
            n_params_tx_cut,
        ) = load_and_cut_transformer_return_pipeline(
            pretrained_name=args.pretrained_model_name_or_path,
            transformer_pretrained=args.transformer_pretrained
            if args.transformer_pretrained is not None
            else args.pretrained_model_name_or_path,
            transformer_quant_type=args.transformer_quant_type,
            cut_transformer_blocks=args.cut_transformer_blocks,
            cut_transformer_blocks_2=args.cut_transformer_blocks_2,
            cut_transformer_components_excluded=args.cut_transformer_components_excluded,
            cut_transformer_type=args.cut_tx_type,
            target_memory_budget=float(args.target_memory_budget)
            if args.target_memory_budget is not None
            else None,
            benchmark_type=args.benchmark_type,
            metric_output_dir=args.metric_output_dir,
            weight_dtype=weight_dtype,
            cache_dir=args.cache_dir,
            debug=args.debug,
        )
    pipe = pipe.to("cuda")

    # Load Prompts
    if args.validation_prompt is None:
        prompts = get_prompts(args.validation_prompts_config)
    else:
        prompts = [args.validation_prompt]

    # Save Config
    seed = np.random.randint(10000) if args.use_rand_seed else args.seed
    full_output_dir_path = Path(args.output_dir).joinpath(
        "_".join(
            [
                datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                (
                    args.transformer_pretrained.split("/")[-2]
                    if args.cut_transformer_blocks is not None
                    else args.pretrained_model_name_or_path.split("/")[-1]
                ),
            ]
        )
    )
    config_gen_img = {
        "args": vars(args),
        "transformer_pretrained": args.transformer_pretrained,
        "transformer_quant_type": args.transformer_quant_type,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "full_output_dir_path": str(full_output_dir_path),
        "prompts": prompts,
        "height": height,
        "width": width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "cut_blocks_l": cut_blocks_l,
        "n_params_transformer_original": n_params_tx_orig,
        "n_params_transformer_distilled": n_params_tx_cut,
        "env_seed": args.seed,
        "infer_seed": seed,
    }
    full_output_dir_path.joinpath("images").mkdir(parents=True, exist_ok=True)
    with open(full_output_dir_path.joinpath("config.json"), "w") as f:
        json.dump(config_gen_img, f, indent=4)
    if args.debug:
        print(f"[INFO] Saving a config file to {full_output_dir_path.joinpath('config.json')}")

    # Run Inference over prompts
    for idx, prompt in enumerate(prompts):
        image = pipe(
            prompt,
            negative_prompt="",
            height=height,
            width=width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        # Save Image
        # When the prompt length is too long, simply save an image with a given image idx.
        try:
            image.save(full_output_dir_path.joinpath("images", f"seed{seed}_{prompt}.png"))
        except:
            image.save(full_output_dir_path.joinpath("images", f"seed{seed}_{idx}.png"))
        if args.debug:
            print(
                f"[INFO] Saving an image to {full_output_dir_path.joinpath('images', f'seed{seed}_{prompt}.png')}"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
