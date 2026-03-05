"""[EXPERIMENTAL] Inference code for SD3 and Flux models with DC-AE VAE replacement.

Generates images using a pruned SD3 transformer paired with a DC-AE autoencoder
instead of the standard VAE, potentially enabling higher-compression latent representations.
"""

import argparse
import datetime
import os
import random
from pathlib import Path
from pprint import pformat

import numpy as np
import safetensors
import torch
from diffusers import SD3Transformer2DModel

from model.builder import get_vae
from pipelines.pipeline_dcae_stable_diffusion_3 import DCAEStableDiffusion3Pipeline
from utils.identity_block import Identity_Block_SD
from utils.utils import (
    adjust_SD3Transformer2DModel_inout4dcae,
    get_num_params,
    parse_cut_blocks,
    return_none,
)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--vae_name",
        type=str,
        default="sd3.5",
        choices=[
            "sd3.5",
            "dc-ae",
        ],
        required=False,
        help="VAE name.",
    )
    parser.add_argument(
        "--vae_pretrained",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        choices=[
            "stabilityai/stable-diffusion-3.5-medium",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "mit-han-lab/dc-ae-f32c32-sana-1.0",
        ],
        required=False,
        help="Path to pretrained VAE model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_pretrained",
        type=return_none,
        default=None,
        required=False,
        help="Path to pretrained Transformer model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        choices=[
            "stabilityai/stable-diffusion-3.5-medium",
            "stabilityai/stable-diffusion-3-medium-diffusers",
        ],
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
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
        "--transformer_patch_size",
        type=int,
        default=1,
        help=("patch size of the transformer."),
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

    # Load VAE
    vae = get_vae(
        name=args.vae_name,
        model_path=args.vae_pretrained,
        dtype=weight_dtype,
        device="cuda",
        cache_dir=args.cache_dir,
    )
    if args.debug:
        print(
            f"[INFO] Num Params of the {args.vae_name} VAE model {get_num_params(vae)/1000**3:0.3f}B"
        )

    # Load Transformer
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        cache_dir=args.cache_dir,
    )
    n_params_tx_orig = get_num_params(transformer)
    if args.debug:
        print(
            f"[INFO] Num Params of the ORIGINAL Transformer model {n_params_tx_orig/1000**3:0.3f}B"
        )

    ##### Modify Transformer In/Out Layers #####
    if args.vae_name == "dc-ae":
        adjust_SD3Transformer2DModel_inout4dcae(
            transformer, args.resolution, vae, args.transformer_patch_size
        )

    # Load distilled/cut transformer blocks.
    cut_blocks_l = parse_cut_blocks(args.cut_transformer_blocks)
    if args.transformer_pretrained is not None:
        # Remove transformer blocks if cut_transformer_blocks are given by a user
        if cut_blocks_l:
            if args.debug:
                print(f"[INFO] Test model with removed transformer blocks {cut_blocks_l}")
            for i in cut_blocks_l:
                transformer.transformer_blocks[i] = Identity_Block_SD()
        else:
            if args.debug:
                print(f"[INFO] Test original model without any cutting.")

        n_params_tx_cut = get_num_params(transformer)
        if args.debug:
            print(
                f"[INFO] Num Params of the DISTILLED model {n_params_tx_cut/1000**3:0.3f} ({n_params_tx_cut/n_params_tx_orig*100:0.3f}% of the original model)"
            )

        # Load distilled/cut transformer blocks checkpoint
        if "checkpoint" in args.transformer_pretrained:
            model_data_path = os.path.join(args.transformer_pretrained, "model.safetensors")
        else:
            model_data_path = os.path.join(
                args.transformer_pretrained,
                "transformer/diffusion_pytorch_model.safetensors",
            )
        model_data = safetensors.torch.load_file(model_data_path)
        transformer.load_state_dict(model_data, strict=False)
    else:
        n_params_tx_cut = n_params_tx_orig
        if args.debug:
            print(f"[INFO] Test original model without any cutting.")

    vae.to("cuda", dtype=weight_dtype)
    transformer.to("cuda", dtype=weight_dtype)

    # Initialise pipeline and load other components
    if args.vae_name == "dc-ae":
        pipe = DCAEStableDiffusion3Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=None,
            transformer=transformer,
            torch_dtype=weight_dtype,
            cache_dir=args.cache_dir,
        )
        pipe.vae = vae
        pipe.vae_scale_factor = 2 ** sum(tmp > 0 for tmp in vae.encoder.cfg.depth_list[:-1])
    else:
        pipe = DCAEStableDiffusion3Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            transformer=transformer,
            torch_dtype=weight_dtype,
            cache_dir=args.cache_dir,
        )
    pipe = pipe.to("cuda")

    seed = np.random.randint(10000) if args.use_rand_seed else args.seed
    # Run Inference
    # TODO: add metrics evaluation with more prompts (not hard-coded prompts).
    # HPSv2: image_dir/model_id/{prompt_type}/{idx:05d}.jpg or .png
    #
    prompt = "A capybara holding a sign that reads Hello Fast World"
    image = pipe(
        prompt,
        negative_prompt="",
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    # Save Config
    full_output_dir_path = Path(args.output_dir).joinpath(
        "_".join(
            [
                datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                (
                    args.transformer_pretrained.split("/")[-2]
                    if args.transformer_pretrained is not None
                    else args.pretrained_model_name_or_path.split("/")[-1]
                ),
                args.vae_name,
            ]
        )
    )
    config_gen_img = {
        "args": args,
        "vae_name": args.vae_name,
        "vae_pretrained": args.vae_pretrained,
        "transformer_pretrained": args.transformer_pretrained,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "full_output_dir_path": str(full_output_dir_path),
        "prompt": prompt,
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
    with open(full_output_dir_path.joinpath("config.txt"), "w") as f:
        f.write(pformat(config_gen_img, sort_dicts=False))
    if args.debug:
        print(f"[INFO] Saving a config file to {full_output_dir_path.joinpath('config.txt')}")

    # Save Image
    image.save(full_output_dir_path.joinpath("images", f"seed{seed}_{prompt}.png"))
    if args.debug:
        print(
            f"[INFO] Saving an image to {full_output_dir_path.joinpath('images', f'seed{seed}_{prompt}.png')}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
