import argparse
import gc
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.builder import load_and_cut_transformer
from profilers.utils import display_memory_peak
from utils.quantize import quantize_linear_layers
from utils.utils import return_none


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Profile memory usage during inference for SD3 transformer models."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3.5-large-turbo",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        "--log_dir",
        type=str,
        default=f"./log/SD35_large_turbo",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="If quantize the linear layers in the transformer model with bitsandbytes.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):

    print(
        f"[INFO] Before loading transformer, allocated memory {torch.cuda.memory_allocated(0)/1024**3}"
    )

    (
        transformer,
        cut_blocks_l,
        cut_fg_dict,
        n_params_tx_orig,
        n_params_tx_cut,
    ) = load_and_cut_transformer(
        pretrained_name=args.pretrained_model_name_or_path,
        cut_transformer_blocks=args.cut_transformer_blocks,
        cut_transformer_blocks_2=None,  # "16-19,30,36",
        cut_transformer_components_excluded="attn,w_norm1",
        cut_transformer_type="cut_blk_manual",
        target_memory_budget=0.71,
        benchmark_type="hpsv2",
        metric_output_dir=f"assets/precomputed_metrics/hpsv2/stable-diffusion-3.5-large-turbo.json",
        debug=True,
    )

    if args.quantize:
        quantize_linear_layers(transformer)

    transformer = transformer.to("cuda:0")

    print(
        f"[INFO] After loading transformer, allocated memory {torch.cuda.memory_allocated(0)/1024**3}"
    )
    # A number of input numbers
    cfg_dim = 1 if "turbo" in args.pretrained_model_name_or_path else 2
    latent_model_input = torch.rand([cfg_dim, 16, 128, 128], dtype=torch.bfloat16).to("cuda:0")
    timestep = torch.rand([cfg_dim], dtype=torch.bfloat16).to("cuda:0")
    prompt_embeds = torch.rand([cfg_dim, 333, 4096], dtype=torch.bfloat16).to("cuda:0")
    pooled_prompt_embeds = torch.rand([cfg_dim, 2048], dtype=torch.bfloat16).to("cuda:0")
    print(
        f"[INFO] Before running transformer, allocated memory {torch.cuda.memory_allocated(0)/1024**3}"
    )
    print(
        f"[INFO] Before running transformer, MAX allocated memory {torch.cuda.max_memory_allocated(0)/1024**3}"
    )

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0.0, active=100, repeat=10),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.log_dir),
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
    print(
        f"[INFO] After running transformer, allocated memory {torch.cuda.memory_allocated(0)/1024**3}"
    )
    print(
        f"[INFO] After running transformer, MAX allocated memory {torch.cuda.max_memory_allocated(0)/1024**3}"
    )

    peak, memory = display_memory_peak(args.log_dir)
    plt.plot(memory)
    del transformer
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_max_memory_allocated()

    print(
        f"[INFO] After removing transformer, allocated memory {torch.cuda.memory_allocated(0)/1024**3}"
    )
    print(
        f"[INFO] After removing transformer, MAX allocated memory {torch.cuda.max_memory_allocated(0)/1024**3}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
