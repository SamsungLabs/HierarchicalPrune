"""Shared utilities for model layer replacement and pruning helpers functions."""

import argparse
import copy
import gc
import json
import runpy
import shlex
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from bitsandbytes.nn import Params4bit
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.models.embeddings import PatchEmbed
from diffusers.utils import is_wandb_available

from utils.attrdict_wrapper import AttrDict

if is_wandb_available():
    import wandb

import glob
from typing import List

from model.dc_ae.efficientvit.models.efficientvit.dc_ae import DCAE


def run_python(target, new_args):
    sys.argv = [target] + shlex.split(new_args)
    runpy.run_path(target, run_name="__main__")


def return_none(arg):
    """Convert ``"none"`` or empty string to ``None``, otherwise pass through."""
    return None if "none" == arg.lower() or "" == arg else arg


def list_of_strings(arg):
    """Parse a newline-separated string into a list of stripped strings."""
    if "\n" in arg:
        old_list = arg.split("\n")
        new_list = []
        for prompt in old_list:
            new_list.append(prompt.strip())
        return new_list
    else:
        return arg.strip()


def summarize_tensor(x):
    """Return a colored string with shape, min, mean, and max of a tensor."""
    return f"\033[34m{str(tuple(x.shape)).ljust(24)}\033[0m (\033[31mmin {x.min().item():+.4f}\033[0m / \033[32mmean {x.mean().item():+.4f}\033[0m / \033[33mmax {x.max().item():+.4f}\033[0m)"


def get_num_params(model):
    """Count total parameters, doubling NF4-quantized parameters to reflect original size."""
    sum = 0
    for p in model.parameters():
        # if parameter p is a quantised parameter, double the count of the number of parameters
        if isinstance(p, Params4bit):
            sum += p.numel() * 2
        else:
            sum += p.numel()
    return sum


def get_num_params_trainable(model):
    """Count trainable parameters, doubling NF4-quantized parameters to reflect original size."""
    sum = 0
    for p in model.parameters():
        if p.requires_grad:
            # if parameter p is a quantised parameter, double the count of the number of parameters
            if isinstance(p, Params4bit):
                sum += p.numel() * 2
            else:
                sum += p.numel()
    return sum


def get_precomputed_metric_scores(
    name_pretrained_l: list,
    metric_output_dir: str,
    benchmark_type_l: list = ["hpsv2"],
):
    """Load precomputed contribution analysis scores from disk.

    Scans ``results_cont_anal/`` and ``results_cont_anal_fg/`` directories for
    per-layer and fine-grained component benchmark results (e.g., HPSv2 scores).

    Args:
        name_pretrained_l: List of model names to load results for.
        metric_output_dir: Root directory containing contribution analysis outputs.
        benchmark_type_l: List of benchmark types to collect (e.g., ``["hpsv2"]``).

    Returns:
        Nested dict: ``{model_name: {layer_idx: {component: {benchmark: scores}}}}``.
    """
    output_dir = Path(metric_output_dir)
    results_cont_analysis = {}
    for name_pretrained in name_pretrained_l:
        for benchmark_type in benchmark_type_l:
            # "results_cont_anal" is hard-coded for now
            results_path_l = glob.glob(
                str(
                    output_dir.joinpath(
                        "results_cont_anal", benchmark_type, name_pretrained, "*", "config.json"
                    )
                )
            )
            results_path_l = sorted(results_path_l)
            for results_path in results_path_l:
                with open(results_path) as f:
                    layer_str_l = Path(results_path).parent.name.split("cut_blk_")[1].split("_")
                    layer_idx = int(layer_str_l[0])
                    if len(layer_str_l) == 1:  # layer wise contribution analysis
                        layer_type = "all"
                    else:  # finegrained contribution analysis
                        layer_type = "_".join([layer_str_l[i] for i in range(1, len(layer_str_l))])

                    if results_cont_analysis.get(name_pretrained, None) is None:
                        results_cont_analysis[name_pretrained] = {}
                    if results_cont_analysis[name_pretrained].get(layer_idx, None) is None:
                        results_cont_analysis[name_pretrained][layer_idx] = {}
                    if (
                        results_cont_analysis[name_pretrained][layer_idx].get(layer_type, None)
                        is None
                    ):
                        results_cont_analysis[name_pretrained][layer_idx][layer_type] = {}
                    if (
                        results_cont_analysis[name_pretrained][layer_idx][layer_type].get(
                            benchmark_type, None
                        )
                        is None
                    ):
                        results_cont_analysis[name_pretrained][layer_idx][layer_type][
                            benchmark_type
                        ] = {}
                    results_cont_analysis[name_pretrained][layer_idx][layer_type][
                        benchmark_type
                    ] = json.load(f)[benchmark_type]

    for name_pretrained in name_pretrained_l:
        for benchmark_type in benchmark_type_l:
            # "results_cont_anal_fg" is hard-coded for now
            results_path_l = glob.glob(
                str(
                    output_dir.joinpath(
                        "results_cont_anal_fg", benchmark_type, name_pretrained, "*", "config.json"
                    )
                )
            )
            results_path_l = sorted(results_path_l)
            for results_path in results_path_l:
                with open(results_path) as f:
                    layer_str_l = Path(results_path).parent.name.split("cut_blk_")[1].split("_")
                    layer_idx = int(layer_str_l[0])
                    if len(layer_str_l) == 1:  # layer wise contribution analysis
                        layer_type = "all"
                    else:  # finegrained contribution analysis
                        layer_type = "_".join([layer_str_l[i] for i in range(1, len(layer_str_l))])

                    if results_cont_analysis.get(name_pretrained, None) is None:
                        results_cont_analysis[name_pretrained] = {}
                    if results_cont_analysis[name_pretrained].get(layer_idx, None) is None:
                        results_cont_analysis[name_pretrained][layer_idx] = {}
                    if (
                        results_cont_analysis[name_pretrained][layer_idx].get(layer_type, None)
                        is None
                    ):
                        results_cont_analysis[name_pretrained][layer_idx][layer_type] = {}
                    if (
                        results_cont_analysis[name_pretrained][layer_idx][layer_type].get(
                            benchmark_type, None
                        )
                        is None
                    ):
                        results_cont_analysis[name_pretrained][layer_idx][layer_type][
                            benchmark_type
                        ] = {}
                    results_cont_analysis[name_pretrained][layer_idx][layer_type][
                        benchmark_type
                    ] = json.load(f)[benchmark_type]

    # sort the items according to layer_idx so that items are starting from layer_idx=0 incrementally.
    for name_pretrained in name_pretrained_l:
        results_cont_analysis[name_pretrained] = dict(
            sorted(results_cont_analysis[name_pretrained].items())
        )
    return results_cont_analysis


def get_metric_results(
    results: dict,
    name: str = "stable-diffusion-3.5-large-turbo",
    layer_type: str = "all",
    benchmark_type: str = "hpsv2",
    score_type: str = "Overall_Score",
    avg_or_std: str = "Avg",
) -> list:
    """Extract a flat list of scores for a given model/component/benchmark combination."""
    ret_l = []
    for k, v in results[name].items():
        ret_l.append(float(v[layer_type][benchmark_type][score_type][avg_or_std]))
    return ret_l


def get_metric_results_dict(
    results: dict,
    name: str = "stable-diffusion-3.5-large-turbo",
    layer_type_l: list = ["all"],
    benchmark_type: str = "hpsv2",
    score_type: str = "Overall_Score",
    avg_or_std: str = "Avg",
) -> list:
    """Build a dict mapping ``"block_idx<tab>component"`` to scores across layer types."""
    ret_dict = {}
    for layer_type in layer_type_l:
        for k, v in results[name].items():
            if v.get(layer_type, None) is None:
                # print(f"[DEBUG] Skipped {layer_type} as it does not exist")
                continue
            ret_dict["<tab>".join([str(k), layer_type])] = float(
                v[layer_type][benchmark_type][score_type][avg_or_std]
            )
    return ret_dict


def get_ranked_transformer_components(
    pretrained_name: str,
    metric_output_dir: str,
    benchmark_type: str = "hpsv2",
    fine_grained_cont_analysis: bool = True,
    debug: bool = True,
):
    """Rank transformer components by contribution score (descending).

    Returns:
        Ordered dict of ``"block_idx<tab>component"`` to score, highest first.
    """
    # get layer type
    layer_type_l = get_transformer_block_components_w_norm(
        pretrained_name=pretrained_name, fine_grained_cont_analysis=fine_grained_cont_analysis
    )

    if "/" in pretrained_name:
        tmp_pretrained_name = pretrained_name.split("/")[-1]

    results_cont_analysis = get_precomputed_metric_scores(
        name_pretrained_l=[tmp_pretrained_name],
        metric_output_dir=metric_output_dir,
        benchmark_type_l=[benchmark_type],
    )

    ret_dict = get_metric_results_dict(
        results=results_cont_analysis,
        name=tmp_pretrained_name,
        layer_type_l=layer_type_l,
        benchmark_type=benchmark_type,
    )
    sorted_ret_dict = dict(sorted(ret_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_ret_dict


def get_transformer_and_pipeline_class(pretrained_name: str):
    """Return the appropriate (TransformerClass, PipelineClass) for the given model."""
    # Determine SD3-series model or Flux-series model
    if "stabilityai/stable-diffusion-3" in pretrained_name:
        TransformerClass = SD3Transformer2DModel
        PipelineClass = StableDiffusion3Pipeline
    elif "black-forest-labs/FLUX.1" in pretrained_name:
        TransformerClass = FluxTransformer2DModel
        PipelineClass = FluxPipeline
    else:
        raise ValueError(
            f"SD3-series or FLUX.1-series models are supported. However, {pretrained_name} is given."
        )
    return TransformerClass, PipelineClass


def get_transformer_block_components_w_norm(
    pretrained_name: str,
    fine_grained_cont_analysis: bool,
) -> List:
    """Return the list of pruneable component names."""
    if not fine_grained_cont_analysis:
        tx_block_components = ["all"]
    else:
        # Determine SD3-series model or Flux-series model
        if "stabilityai/stable-diffusion-3" in pretrained_name:
            tx_block_components = [
                "w_norm1",
                "w_norm1_context",
                "attn",
                "ff",
                "ff_context",
            ]
        elif "black-forest-labs/FLUX.1" in pretrained_name:
            tx_block_components = [
                "w_norm1",
                "w_norm1_context",
                "attn",
                "ff",
                "ff_context",
                "w_norm",
                "proj_mlp_out",
            ]
        else:
            raise ValueError(
                f"SD3-series or FLUX.1-series models are supported. However, {pretrained_name} is given."
            )
    return tx_block_components


def get_transformer_block_components(
    pretrained_name: str,
    fine_grained_cont_analysis: bool,
    fine_grained_cont_analysis_type: str = None,
    tx_block_idx: int = None,
    num_transformer_blocks_wo_single_blocks: int = None,
) -> List:
    """Return the list of pruneable component names for contribution analysis.

    Supports multiple analysis granularities (``"fg"``, ``"hybrid_1"``, ``"hybrid_2"``)
    and differentiates between Flux dual-stream and single-stream blocks.
    """
    if not fine_grained_cont_analysis:
        tx_block_components = ["all"]
    else:
        # Determine SD3-series model or Flux-series model
        if "stabilityai/stable-diffusion-3" in pretrained_name or (
            "black-forest-labs/FLUX.1" in pretrained_name
            and tx_block_idx < num_transformer_blocks_wo_single_blocks
        ):
            if fine_grained_cont_analysis_type is None:
                tx_block_components = [
                    "norm1",
                    "norm1_context",
                    "w_norm1",
                    "w_norm1_context",
                    "attn",
                    "ff",
                    "ff_context",
                ]
            if fine_grained_cont_analysis_type == "fg":
                tx_block_components = [
                    "w_norm1",
                    "w_norm1_context",
                    "attn",
                    "ff",
                    "ff_context",
                ]
            elif fine_grained_cont_analysis_type == "hybrid_1":
                tx_block_components = [
                    "w_norm1+w_norm1_context",
                    "w_norm1+ff",
                    "w_norm1_context+ff_context",
                ]
            elif fine_grained_cont_analysis_type == "hybrid_2":
                tx_block_components = [
                    "ff_context+ff",
                    "w_norm1_context+ff_context+ff",
                    "w_norm1+w_norm1_context+ff_context+ff",
                ]
        elif (
            "black-forest-labs/FLUX.1" in pretrained_name
            and tx_block_idx >= num_transformer_blocks_wo_single_blocks
        ):
            if fine_grained_cont_analysis_type is None:
                tx_block_components = ["norm", "w_norm", "attn", "proj_mlp_out"]
            if fine_grained_cont_analysis_type == "fg":
                tx_block_components = [
                    "w_norm",
                    "attn",
                    "proj_mlp_out",
                ]
            elif fine_grained_cont_analysis_type == "hybrid_1":
                tx_block_components = [
                    "w_norm+attn",
                    "w_norm+proj_mlp_out",
                    "attn+proj_mlp_out",
                ]
            # if tx_block_idx < num_transformer_blocks_wo_single_blocks:  # FluxTransformerBlock
            #     # this case is considered above (leave this if statement here to make it clear).
            #     pass
            # else:  # FluxTransformerBlock + FluxSingleTransformerBlock
            #     tx_block_components = ["norm", "w_norm", "attn", "proj_mlp_out"]
        else:
            raise ValueError(
                f"SD3-series or FLUX.1-series models are supported. However, {pretrained_name} is given."
            )
    return tx_block_components


def get_num_transformer_blocks(pretrained_name: str, cache_dir: str = None):
    """Load a transformer to count the number of total blocks and its dual-stream blocks."""
    # Determine SD3-series model or Flux-series model
    TransformerClass, _ = get_transformer_and_pipeline_class(pretrained_name=pretrained_name)

    transformer = TransformerClass.from_pretrained(
        pretrained_name, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=cache_dir
    )
    num_transformer_blocks = 0
    num_transformer_blocks_wo_single_blocks = 0
    # Both SD3 and Flux models have transformer_blocks
    if hasattr(transformer, "transformer_blocks"):
        num_transformer_blocks += len(transformer.transformer_blocks)
        num_transformer_blocks_wo_single_blocks += len(transformer.transformer_blocks)
    # Only Flux models have single_transformer_blocks
    if hasattr(transformer, "single_transformer_blocks"):
        num_transformer_blocks += len(transformer.single_transformer_blocks)
    assert (
        num_transformer_blocks > 0
    ), f"num_transformer_blocks=={num_transformer_blocks} but it is supposed to be larger than 0."

    del transformer
    return num_transformer_blocks, num_transformer_blocks_wo_single_blocks


def get_prompts(config_path):
    """Load evaluation prompts from a JSON or YAML config file."""
    if ".json" in config_path:
        with open(config_path) as f:
            prompts = json.load(f)
        prompts_list = prompts.keys()
    elif ".yaml" in config_path or ".yml" in config_path:
        with open(config_path) as f:
            prompts = yaml.load(f, Loader=yaml.FullLoader)
            prompts = AttrDict(prompts)
        prompts_list = []
        for _, v in prompts.items():
            prompts_list.extend(v)
    else:
        prompts_list = None

    return prompts_list


# Parse the cut transformer blocks argument from "1,2" or "1-2" or "1,2-4,6-8" to "[1,2]", "[1,2]", "[1,2,3,4,6,7,8]"
# if None is given, return an empty list so that there will be no removal of transformer blocks.
def parse_cut_blocks(cut_transformer_blocks):
    if cut_transformer_blocks is not None:
        tmp_cut_blocks_l = cut_transformer_blocks.split(",")
        cut_blocks_l = []
        for cut_blocks_str in tmp_cut_blocks_l:
            if "-" in cut_blocks_str:
                cut_blocks = list(
                    range(
                        int(cut_blocks_str.split("-")[0]),
                        int(cut_blocks_str.split("-")[1]),
                    )
                )
            else:
                cut_blocks = [int(cut_blocks_str)]
            cut_blocks_l += cut_blocks
        return cut_blocks_l
    else:
        return []


def log_validation(
    pipeline,
    pipeline_args,
    global_step,
    args,
    device="cuda",
    trackers=None,
    is_final_validation=False,
):
    """Run inference on validation prompts and log images to TensorBoard/W&B."""
    pipeline = pipeline.to(device)
    # pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed else None
    prompts = pipeline_args.pop("prompt")

    for p_idx, prompt in enumerate(prompts):
        with torch.autocast(device_type=device, dtype=pipeline.dtype):
            images = [
                pipeline(prompt=prompt, **pipeline_args, generator=generator).images[0]
                for _ in range(args.num_validation_images)
            ]

        phase_name = "Test" if is_final_validation else "Validation"
        if args.save_validation_images:
            for i_idx, img in enumerate(images):
                img.save(
                    str(
                        Path(
                            args.output_dir,
                            args.logging_dir,
                            f"{phase_name}_gstep{global_step}_prompt{p_idx}_img{i_idx}.png",
                        ).resolve()
                    )
                )

        if trackers is not None:
            for tracker in trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        f"{phase_name} Prompt {p_idx}: {prompt}",
                        np_images,
                        global_step,
                        dataformats="NHWC",
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            phase_name: [
                                wandb.Image(image, caption=f"Prompt {p_idx}: {prompt}")
                                for i_idx, image in enumerate(images)
                            ]
                        }
                    )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    return images


def adjust_SD3Transformer2DModel_inout4dcae(
    transformer: SD3Transformer2DModel,
    resolution: int = 1024,
    ae: DCAE = None,
    patch_size: int = 1,
    init_type: str = None,
    return_params: bool = False,
):
    """[EXPERIMENTAL] Adapt the SD3 transformer's input/output projections for DC-AE latent dimensions.

    Replaces ``pos_embed`` and ``proj_out`` layers to match DC-AE's channel count and
    spatial resolution. Supports multiple weight initialization strategies:
    ``"all_rand"``, ``"partial_zeros"``, ``"partial_rand"``, ``"partial_repeat"``.

    Args:
        transformer: The SD3 transformer model to modify in-place.
        resolution: Target image resolution.
        ae: Optional DC-AE model (used to infer latent channels and scale factor).
        patch_size: Patch size for the new ``PatchEmbed`` layer.
        init_type: Weight initialization strategy for the new layers.
        return_params: If True, also return the original layer state dicts.
    """
    # Transformer input layer modification
    inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

    # patch_size = 1  # hard coded based on DC-AE f32c32
    if ae is not None:
        in_channels = out_channels = ae.cfg.latent_channels
        vae_scale_factor = 2 ** sum(tmp > 0 for tmp in ae.encoder.cfg.depth_list[:-1])
    else:  # hard coded based on DC-AE f32c32
        in_channels = out_channels = 32
        vae_scale_factor = 32

    orig_pos_embed_data = transformer.pos_embed.state_dict()
    transformer.pos_embed = PatchEmbed(
        height=resolution // vae_scale_factor,
        width=resolution // vae_scale_factor,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=inner_dim,
        pos_embed_max_size=transformer.config.pos_embed_max_size,
    )

    # Transformer output layer modification
    orig_proj_out_data = transformer.proj_out.state_dict()
    transformer.proj_out = torch.nn.Linear(
        inner_dim, patch_size * patch_size * out_channels, bias=True
    )

    if init_type is not None:
        with torch.no_grad():
            if init_type == "all_rand":
                pass
            elif init_type == "partial_zeros":
                if patch_size == 2:
                    torch.nn.init.zeros_(
                        transformer.pos_embed.proj.weight
                    )  # [1536, 16, 2, 2] => [1536, 32, 2, 2]
                    # torch.nn.init.zeros_(transformer.pos_embed.proj.bias) # [1536] => [1536] dim doesn't change, thus no need to init with zeros
                    transformer.pos_embed.proj.weight[
                        :, : orig_pos_embed_data["proj.weight"].shape[1]
                    ] = orig_pos_embed_data["proj.weight"]
                    transformer.pos_embed.proj.bias[:] = orig_pos_embed_data["proj.bias"]

                    torch.nn.init.zeros_(transformer.proj_out.weight)  # [64, 1536] => [128, 1536]
                    torch.nn.init.zeros_(transformer.proj_out.bias)  # [64] => [128]
                    transformer.proj_out.weight[
                        : orig_proj_out_data["weight"].shape[0], :
                    ] = orig_proj_out_data["weight"]
                    transformer.proj_out.bias[
                        : orig_proj_out_data["bias"].shape[0]
                    ] = orig_proj_out_data["bias"]
                elif patch_size == 1:
                    torch.nn.init.zeros_(
                        transformer.pos_embed.proj.weight
                    )  # [1536, 16, 2, 2] => [1536, 32, 1, 1]
                    # transformer.pos_embed.proj.bias: [1536] => [1536]
                    orig_pos_embed_data["proj.weight"] = (
                        orig_pos_embed_data["proj.weight"]
                        .mean(2, keepdims=True)
                        .mean(3, keepdims=True)
                    )
                    transformer.pos_embed.proj.weight[
                        :, : orig_pos_embed_data["proj.weight"].shape[1]
                    ] = orig_pos_embed_data["proj.weight"]
                    transformer.pos_embed.proj.bias[:] = orig_pos_embed_data["proj.bias"]

                    w_shape = (
                        transformer.proj_out.weight.shape
                    )  # [64, 1536] => [32, 1536] # dim decreases, thus no need to init with zeros
                    b_shape = transformer.proj_out.bias.shape  # [64] => [32]
                    transformer.proj_out.weight[:] = (
                        orig_proj_out_data["weight"][: w_shape[0]]
                        + orig_proj_out_data["weight"][w_shape[0] :]
                    ) / 2
                    transformer.proj_out.bias[:] = (
                        orig_proj_out_data["bias"][: b_shape[0]]
                        + orig_proj_out_data["bias"][b_shape[0] :]
                    ) / 2
                else:
                    raise NotImplementedError()
            elif init_type == "partial_rand":
                # By default, weights are initialised randomly (e.g., nn.init.kaiming_uniform_), thus separate init function is not needed here
                if patch_size == 2:
                    transformer.pos_embed.proj.weight[
                        :, : orig_pos_embed_data["proj.weight"].shape[1]
                    ] = orig_pos_embed_data["proj.weight"]
                    transformer.pos_embed.proj.bias[:] = orig_pos_embed_data["proj.bias"]

                    transformer.proj_out.weight[
                        : orig_proj_out_data["weight"].shape[0], :
                    ] = orig_proj_out_data["weight"]
                    transformer.proj_out.bias[
                        : orig_proj_out_data["bias"].shape[0]
                    ] = orig_proj_out_data["bias"]
                elif patch_size == 1:
                    orig_pos_embed_data["proj.weight"] = (
                        orig_pos_embed_data["proj.weight"]
                        .mean(2, keepdims=True)
                        .mean(3, keepdims=True)
                    )
                    transformer.pos_embed.proj.weight[
                        :, : orig_pos_embed_data["proj.weight"].shape[1]
                    ] = orig_pos_embed_data["proj.weight"]
                    transformer.pos_embed.proj.bias[:] = orig_pos_embed_data["proj.bias"]

                    w_shape = transformer.proj_out.weight.shape
                    b_shape = transformer.proj_out.bias.shape
                    transformer.proj_out.weight[:] = (
                        orig_proj_out_data["weight"][: w_shape[0]]
                        + orig_proj_out_data["weight"][w_shape[0] :]
                    ) / 2
                    transformer.proj_out.bias[:] = (
                        orig_proj_out_data["bias"][: b_shape[0]]
                        + orig_proj_out_data["bias"][b_shape[0] :]
                    ) / 2
                else:
                    raise NotImplementedError()
            elif init_type == "partial_repeat":
                if patch_size == 2:
                    # transformer.pos_embed.proj.weight: [1536, 16, 2, 2] => [1536, 32, 2, 2]
                    # transformer.pos_embed.proj.bias: [1536] => [1536] dim doesn't change, thus no need to init by repeating
                    transformer.pos_embed.proj.weight[
                        :, : orig_pos_embed_data["proj.weight"].shape[1]
                    ] = orig_pos_embed_data["proj.weight"]
                    transformer.pos_embed.proj.weight[
                        :, orig_pos_embed_data["proj.weight"].shape[1] :
                    ] = orig_pos_embed_data["proj.weight"]
                    transformer.pos_embed.proj.bias[:] = orig_pos_embed_data["proj.bias"]

                    # transformer.proj_out.weight: [64, 1536] => [128, 1536]
                    # transformer.proj_out.bias: [64] => [128]
                    transformer.proj_out.weight[
                        : orig_proj_out_data["weight"].shape[0], :
                    ] = orig_proj_out_data["weight"]
                    transformer.proj_out.weight[
                        orig_proj_out_data["weight"].shape[0] :, :
                    ] = orig_proj_out_data["weight"]
                    transformer.proj_out.bias[
                        : orig_proj_out_data["bias"].shape[0]
                    ] = orig_proj_out_data["bias"]
                    transformer.proj_out.bias[
                        orig_proj_out_data["bias"].shape[0] :
                    ] = orig_proj_out_data["bias"]
                elif patch_size == 1:
                    # transformer.pos_embed.proj.weight: [1536, 16, 2, 2] => [1536, 32, 1, 1]
                    # transformer.pos_embed.proj.bias: [1536] => [1536]
                    orig_pos_embed_data["proj.weight"] = (
                        orig_pos_embed_data["proj.weight"]
                        .mean(2, keepdims=True)
                        .mean(3, keepdims=True)
                    )
                    transformer.pos_embed.proj.weight[
                        :, : orig_pos_embed_data["proj.weight"].shape[1]
                    ] = orig_pos_embed_data["proj.weight"]
                    transformer.pos_embed.proj.weight[
                        :, orig_pos_embed_data["proj.weight"].shape[1] :
                    ] = orig_pos_embed_data["proj.weight"]
                    transformer.pos_embed.proj.bias[:] = orig_pos_embed_data["proj.bias"]

                    w_shape = (
                        transformer.proj_out.weight.shape
                    )  # [64, 1536] => [32, 1536] # dim decreases, thus no need to init by repeating
                    b_shape = transformer.proj_out.bias.shape  # [64] => [32]
                    transformer.proj_out.weight[:] = (
                        orig_proj_out_data["weight"][: w_shape[0]]
                        + orig_proj_out_data["weight"][w_shape[0] :]
                    ) / 2
                    transformer.proj_out.bias[:] = (
                        orig_proj_out_data["bias"][: b_shape[0]]
                        + orig_proj_out_data["bias"][b_shape[0] :]
                    ) / 2
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    # reinit config of transformer
    transformer.config.sample_size = resolution // vae_scale_factor
    transformer.patch_size = patch_size
    transformer.in_channels = in_channels
    transformer.out_channels = out_channels
    transformer.config.patch_size = patch_size
    transformer.config.in_channels = in_channels
    transformer.config.out_channels = out_channels

    if return_params:
        return transformer, orig_pos_embed_data, orig_proj_out_data


if __name__ == "__main__":
    # Test parse_cut_blocks
    print(f"\n=============================================================")
    print(f"[INFO] Test parse_cut_blocks")
    print(parse_cut_blocks("1,2,3,4,5"))
    print(parse_cut_blocks("1-5"))
    print(parse_cut_blocks("1,2-5"))
    print(parse_cut_blocks("1-2,3-5"))
    print(parse_cut_blocks("1-2,5-4"))

    # Test log_validation
    print(f"\n=============================================================")
    print(f"[INFO] Test log_validation")
    args = argparse.ArgumentParser(description="Simple example of a argparse.").parse_args()
    args.seed = 1234  # hard-coded for a testing purpose
    args.num_validation_images = 2
    args.output_dir = "../distil_Diffusers_ckpts"
    args.logging_dir = "logs/"
    args.validation_prompts = ["A cyberpunk cat with a sign reading Hello Fast World.", "A dog."]
    args.resolution = 1024
    args.num_inference_steps = 28
    args.guidance_scale = 7.0
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium"
    args.cache_dir = None
    args.save_validation_images = True
    Path(args.output_dir, args.logging_dir).mkdir(parents=True, exist_ok=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
    # args as an input for pipeline
    pipeline_args = {
        "prompt": args.validation_prompts,
        "negative_prompt": "",
        "height": args.resolution,
        "width": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
    }
    log_validation(
        pipeline,
        pipeline_args=pipeline_args,
        global_step=100,
        args=args,
        device="cuda",
        trackers=None,
        is_final_validation=False,
    )

    # Test init type of adjust_SD3Transformer2DModel_inout4dcae
    print(f"\n=============================================================")
    print(f"[INFO] Test init type of adjust_SD3Transformer2DModel_inout4dcae")
    transformer = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",  # hard-coded for a testing purpose
        subfolder="transformer",
        cache_dir=None,
    )
    patch_size_l = [1, 2]
    init_type_l = ["all_rand", "partial_zeros", "partial_rand", "partial_repeat"]
    for init_type in init_type_l:
        for patch_size in patch_size_l:
            print(f"\n[INFO] Testing init_type: {init_type}, patch_size: {patch_size}")
            (
                new_transformer,
                orig_pos_embed_data,
                orig_proj_out_data,
            ) = adjust_SD3Transformer2DModel_inout4dcae(
                copy.deepcopy(transformer),
                resolution=1024,
                ae=None,
                patch_size=patch_size,
                init_type=init_type,
                return_params=True,
            )
            if init_type == "partial_repeat":
                print(f"For partial_repeat, we check if first few elements are repeated")
                print(
                    f"Original pos_embed_data: {orig_pos_embed_data['proj.weight'][0,:5,0,0].detach().cpu().numpy()}"
                )
                print(
                    f"New      pos_embed_data: {new_transformer.pos_embed.proj.weight[0,:5,0,0].detach().cpu().numpy()}"
                )
            print(
                f"Original pos_embed_data: {orig_pos_embed_data['proj.weight'][0,10:20,0,0].detach().cpu().numpy()}"
            )
            print(
                f"New      pos_embed_data: {new_transformer.pos_embed.proj.weight[0,10:20,0,0].detach().cpu().numpy()}"
            )
            del new_transformer
