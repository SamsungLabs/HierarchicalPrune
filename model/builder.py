"""Build and configure pruned transformer models.

Provides utilities for loading pretrained SD3/Flux transformers, applying hierarchical
pruning (block-level and fine-grained component removal), VAE encoding/decoding
(standard and DC-AE variants), and optional NF4 quantization.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import PIL
import safetensors
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.models import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
from diffusers.utils import PIL_INTERPOLATION
from PIL import Image, ImageOps

from model.dc_ae.efficientvit.ae_model_zoo import DCAE_HF
from model.transformers import forward_FluxSingleTransformerBlock
from utils.identity_block import (
    Identity_Block_AdaLayerNormZero,
    Identity_Block_AdaLayerNormZeroSingle,
    Identity_Block_FeedForward,
    Identity_Block_Flux,
    Identity_Block_Flux_Single,
    Identity_Block_JointAttn,
    Identity_Block_SD,
)
from utils.quantize import quantize_linear_layers
from utils.utils import (
    get_num_params,
    get_ranked_transformer_components,
    get_transformer_and_pipeline_class,
    parse_cut_blocks,
)


def preprocess(image, device="cuda", dtype=torch.float32):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [
            np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image.to(device).type(dtype)


def get_vae(name, model_path, dtype=torch.float16, device="cuda", cache_dir=None):
    """Load a VAE model (standard AutoencoderKL or DC-AE) by model family name."""
    if name in ["sdxl", "sd3", "sd3.5", "flux"]:
        try:
            vae = (
                AutoencoderKL.from_pretrained(model_path, cache_dir=cache_dir).to(device).to(dtype)
            )
        except:
            vae = (
                AutoencoderKL.from_pretrained(model_path, subfolder="vae", cache_dir=cache_dir)
                .to(device)
                .to(dtype)
            )
        if name == "sdxl":
            vae.config.shift_factor = 0
        return vae
    elif "dc-ae" in name:
        print(f"[DC-AE] Loading model from {model_path}")
        dc_ae = DCAE_HF.from_pretrained(model_path, cache_dir=cache_dir).to(device).to(dtype).eval()
        dc_ae.dtype = dtype
        return dc_ae
    else:
        print("error load vae")
        exit()


def vae_encode(name, vae, images, sample_posterior):
    """Encode images to latent space using the appropriate VAE variant."""
    if name in ["sdxl", "sd3", "sd3.5", "flux"]:
        posterior = vae.encode(images).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    elif "dc-ae" in name:
        ae = vae
        z = ae.encode(images)
        z = z * ae.cfg.scaling_factor
    else:
        print("error load vae")
        exit()
    return z


def vae_decode(name, vae, latent):
    """Decode latent representations back to pixel space."""
    if name in ["sdxl", "sd3", "sd3.5", "flux"]:
        latent = (latent / vae.config.scaling_factor) + vae.config.shift_factor
        samples = vae.decode(latent).sample
    elif "dc-ae" in name:
        ae = vae
        samples = ae.decode(latent / ae.cfg.scaling_factor)
    else:
        print("error load vae")
        exit()
    return samples


def get_DiffusersAPIBitsAndBytesConfig(precision: str, weight_dtype: Any):
    """Create a diffusers ``BitsAndBytesConfig`` for the given precision level."""
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

    if precision == "int4":
        quantization_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=weight_dtype,
        )
    elif precision == "int8":
        quantization_config = DiffusersBitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_TransformersAPIBitsAndBytesConfig(precision: str, weight_dtype: Any):
    """Create a transformers ``BitsAndBytesConfig`` for the given precision level."""
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

    if precision == "int4":
        quantization_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=weight_dtype,
        )
    elif precision == "int8":
        quantization_config = TransformersBitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def cut_transformer_block_component(
    transformer: Union[FluxTransformer2DModel, SD3Transformer2DModel],
    pretrained_name: str,
    cut_blocks_l: List,
    cut_transformer_blocks_component: str,
):
    """Replace specified transformer block components with identity blocks.

    Supports both full block removal (``"all"``) and fine-grained component removal
    (``"attn"``, ``"ff"``, ``"norm1"``, etc.) for SD3 and Flux architectures.

    Args:
        transformer: The transformer model to prune.
        pretrained_name: model name or HuggingFace model ID
        cut_blocks_l: List of block indices to prune.
        cut_transformer_blocks_component: Component(s) to remove, ``"+"``-separated
            for multiple (e.g., ``"attn+ff"``).

    Returns:
        Number of parameters remaining after pruning.
        (The transformer model is updated with in-place operation for the block/component replacement.)
    """
    for i in cut_blocks_l:
        cut_tx_blk_comp_l = cut_transformer_blocks_component.split("+")
        for cut_tx_blk_comp in cut_tx_blk_comp_l:
            if (
                "stabilityai/stable-diffusion-3" in pretrained_name
                or "sd3" in pretrained_name
                or "sd35" in pretrained_name
            ):
                use_dual_attention = transformer.transformer_blocks[i].use_dual_attention
                if cut_tx_blk_comp == "all":
                    transformer.transformer_blocks[i] = Identity_Block_SD()
                elif cut_tx_blk_comp == "norm1":
                    transformer.transformer_blocks[i].norm1 = Identity_Block_AdaLayerNormZero(
                        use_dual_attention=use_dual_attention
                    )
                elif cut_tx_blk_comp == "norm1_context":
                    transformer.transformer_blocks[
                        i
                    ].norm1_context = Identity_Block_AdaLayerNormZero(
                        context_pre_only=transformer.transformer_blocks[i].context_pre_only
                    )
                elif cut_tx_blk_comp == "w_norm1":
                    transformer.transformer_blocks[i].norm1 = Identity_Block_AdaLayerNormZero(
                        embedding_dim=transformer.inner_dim, use_dual_attention=use_dual_attention
                    )
                elif cut_tx_blk_comp == "w_norm1_context":
                    transformer.transformer_blocks[
                        i
                    ].norm1_context = Identity_Block_AdaLayerNormZero(
                        embedding_dim=transformer.inner_dim,
                        context_pre_only=transformer.transformer_blocks[i].context_pre_only,
                    )
                elif cut_tx_blk_comp == "attn":
                    transformer.transformer_blocks[i].attn = Identity_Block_JointAttn()
                    if transformer.transformer_blocks[i].attn2 is not None:
                        transformer.transformer_blocks[i].attn2 = Identity_Block_JointAttn()
                elif cut_tx_blk_comp == "ff":
                    transformer.transformer_blocks[i].ff = Identity_Block_FeedForward()
                elif cut_tx_blk_comp == "ff_context":
                    if transformer.transformer_blocks[i].context_pre_only:
                        assert (
                            transformer.transformer_blocks[i].ff_context is None
                        ), f"The model's ff_context should be initialised with None, but with {transformer.transformer_blocks[i].ff_context}."
                    else:
                        transformer.transformer_blocks[i].ff_context = Identity_Block_FeedForward()

            elif "black-forest-labs/FLUX.1" in pretrained_name or "flux" in pretrained_name:
                if i < transformer.config.num_layers:  # FluxTransformerBlock
                    if cut_tx_blk_comp == "all":
                        transformer.transformer_blocks[i] = Identity_Block_Flux()
                    elif cut_tx_blk_comp == "norm1":
                        transformer.transformer_blocks[i].norm1 = Identity_Block_AdaLayerNormZero()
                    elif cut_tx_blk_comp == "norm1_context":
                        transformer.transformer_blocks[
                            i
                        ].norm1_context = Identity_Block_AdaLayerNormZero()
                    elif cut_tx_blk_comp == "w_norm1":
                        transformer.transformer_blocks[i].norm1 = Identity_Block_AdaLayerNormZero(
                            embedding_dim=transformer.inner_dim,
                        )
                    elif cut_tx_blk_comp == "w_norm1_context":
                        transformer.transformer_blocks[
                            i
                        ].norm1_context = Identity_Block_AdaLayerNormZero(
                            embedding_dim=transformer.inner_dim,
                        )
                    elif cut_tx_blk_comp == "attn":
                        transformer.transformer_blocks[i].attn = Identity_Block_JointAttn()
                    elif cut_tx_blk_comp == "ff":
                        transformer.transformer_blocks[i].ff = Identity_Block_FeedForward()
                    elif cut_tx_blk_comp == "ff_context":
                        transformer.transformer_blocks[i].ff_context = Identity_Block_FeedForward()
                elif (
                    i < transformer.config.num_layers + transformer.config.num_single_layers
                ):  # FluxTransformerBlock + FluxSingleTransformerBlock
                    if cut_tx_blk_comp == "all":
                        transformer.single_transformer_blocks[
                            i - transformer.config.num_layers
                        ] = Identity_Block_Flux_Single()
                    elif cut_tx_blk_comp == "norm":
                        transformer.single_transformer_blocks[
                            i - transformer.config.num_layers
                        ].norm = Identity_Block_AdaLayerNormZeroSingle()
                    elif cut_tx_blk_comp == "w_norm":
                        transformer.single_transformer_blocks[
                            i - transformer.config.num_layers
                        ].norm = Identity_Block_AdaLayerNormZeroSingle(
                            embedding_dim=transformer.inner_dim,
                        )
                    elif cut_tx_blk_comp == "attn":
                        transformer.single_transformer_blocks[
                            i - transformer.config.num_layers
                        ].attn = Identity_Block_JointAttn()
                    elif cut_tx_blk_comp == "proj_mlp_out":
                        if FluxSingleTransformerBlock.forward != forward_FluxSingleTransformerBlock:
                            FluxSingleTransformerBlock.forward = forward_FluxSingleTransformerBlock
                        transformer.single_transformer_blocks[
                            i - transformer.config.num_layers
                        ].proj_mlp = Identity_Block_FeedForward()
                        transformer.single_transformer_blocks[
                            i - transformer.config.num_layers
                        ].proj_out = Identity_Block_FeedForward()
                else:
                    raise ValueError(
                        f"cut_transformer_blocks should be less than {transformer.config.num_layers + transformer.config.num_single_layers}.\
                            However, max(cut_blocks_l)=={max(cut_blocks_l)} is given."
                    )
            else:
                raise ValueError(
                    f"SD3-series or FLUX.1-series models are supported. However, {pretrained_name} is given."
                )
    return get_num_params(transformer)


def cut_transformer_block_component_range(
    pretrained_name: str,
    transformer: Union[FluxTransformer2DModel, SD3Transformer2DModel],
    n_params_tx_orig: int,
    sorted_ret_dict: dict,
    cut_blocks_l: list,
    cut_fg_dict: defaultdict,
    target_memory_budget: float,
    cut_component_except_l: list = None,
    debug: bool = True,
):
    """Iteratively prune components ranked by contribution score until the target memory budget is satisfied.

    Components are removed in order of least contribution (from ``sorted_ret_dict``)
    until the remaining parameter ratio drops below ``target_memory_budget``.

    Args:
        pretrained_name: model name or HuggingFace model ID
        transformer: The transformer model to prune.
        n_params_tx_orig: Original parameter count (for ratio computation).
        sorted_ret_dict: Components ranked by contribution score (descending) (e.g., "31<tab>ff": 30.79).
        cut_blocks_l: Block indices eligible for pruning.
        cut_fg_dict: Accumulator dict mapping block index to list of pruned components.
        target_memory_budget: Target remaining parameter ratio (e.g., 0.7 for 70%).
        cut_component_except_l: Components to exclude from pruning.
        debug: Whether to print pruning progress.
    """
    for k, v in sorted_ret_dict.items():
        # B. Remove one block/component
        cut_blk = int(k.split("<tab>")[0])
        cut_comp = k.split("<tab>")[1]
        if cut_blk not in cut_blocks_l:
            continue

        if cut_component_except_l is not None and cut_comp in cut_component_except_l:
            continue

        cut_fg_dict[cut_blk].append(cut_comp)
        n_params_tx_cut = cut_transformer_block_component(
            transformer=transformer,
            pretrained_name=pretrained_name,
            cut_blocks_l=[cut_blk],
            cut_transformer_blocks_component=cut_comp,
        )

        # C. Check the remained transformer meets the target budgets (e.g., memory)
        if debug:
            print(
                f"[INFO] Pruned {int(k.split('<tab>')[0])}th Layer's {k.split('<tab>')[1]}, score: {v} # Params of Transformer: {n_params_tx_cut/1000**3:0.3f} B, Remaining Ratio: {float(n_params_tx_cut / n_params_tx_orig)*100:0.3f}%, Target Memory Budget: {target_memory_budget*100:0.3f}%"
            )
        if float(n_params_tx_cut / n_params_tx_orig) < target_memory_budget:
            break


def cut_transformer(
    pretrained_name: str,
    transformer: Union[FluxTransformer2DModel, SD3Transformer2DModel],
    cut_transformer_blocks: str = None,
    cut_transformer_blocks_2: str = None,
    cut_transformer_components_excluded: str = None,
    cut_transformer_type: str = "cut_blk_manual",
    target_memory_budget: float = None,
    benchmark_type: str = "hpsv2",
    metric_output_dir: str = None,
    debug: bool = True,
):
    """Apply hierarchical pruning to a transformer model.

    Supports three pruning strategies:
    - ``"cut_blk_manual"``: Remove entire blocks at specified indices.
    - ``"cut_fg_least_drop"``: Remove fine-grained components by contribution score.
    - ``"cut_hybrid"``: Remove entire blocks first, then fine-grained components.

    Args:
        pretrained_name: model name or HuggingFace model ID
        transformer: The transformer model to prune in-place.
        cut_transformer_blocks: Block indices to prune (e.g., ``"38,39"`` or ``"25-30"``).
        cut_transformer_blocks_2: Additional block indices for hybrid pruning stage 2.
        cut_transformer_components_excluded: Components excluded from pruning (e.g., ``"attn,w_norm1"``).
        cut_transformer_type: Pruning strategy as described above.
        target_memory_budget: Target remaining parameter ratio.
        benchmark_type: Metric used for contribution ranking (e.g., ``"hpsv2"``).
        metric_output_dir: (1) Path to precomputed metrics results (``"assets/precomputed_metrics/*/METRIC/MODEL_ID.json"``) or (2) Path to raw output results of contribution analysis (``"results_cont_anal/"``).
        debug: Whether to print debug info.

    Returns:
        Tuple of (list of pruned block indices, dict of pruned components per block).
    """
    n_params_tx_orig = get_num_params(transformer)
    if cut_transformer_type == "cut_blk_manual":
        cut_blocks_l = parse_cut_blocks(cut_transformer_blocks)
        cut_fg_dict = defaultdict(list)
        if cut_blocks_l:
            if debug:
                print(f"[INFO] Test model with removed transformer blocks {cut_blocks_l}")
            n_params_tx_cut = cut_transformer_block_component(
                transformer=transformer,
                pretrained_name=pretrained_name,
                cut_blocks_l=cut_blocks_l,
                cut_transformer_blocks_component="all",
            )
            for cut_blk in cut_blocks_l:
                cut_fg_dict[cut_blk].append("all")
            if debug:
                print(
                    f"[INFO] Num Params of the DISTILLED model {n_params_tx_cut/1000**3:0.3f} B ({n_params_tx_cut/n_params_tx_orig*100:0.3f}% of the original model)"
                )
        else:
            n_params_tx_cut = 0
            if debug:
                print(f"[INFO] Test original model without any cutting.")
    else:
        # A. Get ranked transformer blocks and/or sub-component: {'0<tab>attn': score, ...}
        # Ordered according to the score metric
        cut_blocks_range_l = parse_cut_blocks(cut_transformer_blocks)
        if cut_transformer_components_excluded is not None:
            cut_component_except_l = cut_transformer_components_excluded.split(",")
        else:
            cut_component_except_l = None
        cut_fg_dict = defaultdict(list)
        fine_grained_cont_analysis = False if "cut_blk_" in cut_transformer_type else True
        if Path(metric_output_dir).is_file():
            with open(metric_output_dir) as f:
                sorted_ret_dict = json.load(f)
        else:
            sorted_ret_dict = get_ranked_transformer_components(
                pretrained_name=pretrained_name,
                metric_output_dir=metric_output_dir,
                benchmark_type=benchmark_type,
                fine_grained_cont_analysis=fine_grained_cont_analysis,
            )
        if debug:
            print(f"[INFO] Sorted_ret_dict: {list(sorted_ret_dict.items())[:10]}")

        if cut_transformer_type in ["cut_fg_least_drop", "cut_fg_least_drop_param"]:
            # Step 1: Remove the fine-grained components
            cut_transformer_block_component_range(
                pretrained_name=pretrained_name,
                transformer=transformer,
                n_params_tx_orig=n_params_tx_orig,
                sorted_ret_dict=sorted_ret_dict,
                cut_blocks_l=cut_blocks_range_l,
                cut_component_except_l=cut_component_except_l,
                cut_fg_dict=cut_fg_dict,
                target_memory_budget=target_memory_budget,
                debug=debug,
            )
        elif cut_transformer_type in ["cut_hybrid"]:
            # Step 1: Remove the entire transformer blocks based on "cut_blocks_range_l"
            if cut_blocks_range_l:
                if debug:
                    print(f"[INFO] Test model with removed transformer blocks {cut_blocks_range_l}")
                n_params_tx_cut = cut_transformer_block_component(
                    transformer=transformer,
                    pretrained_name=pretrained_name,
                    cut_blocks_l=cut_blocks_range_l,
                    cut_transformer_blocks_component="all",
                )
                for cut_blk in cut_blocks_range_l:
                    cut_fg_dict[cut_blk].append("all")
                if debug:
                    print(
                        f"[INFO] Num Params of the DISTILLED model {n_params_tx_cut/1000**3:0.3f} B ({n_params_tx_cut/n_params_tx_orig*100:0.3f}% of the original model)"
                    )

            # Step 2: Remove the fine-grained components based on "cut_blocks_range_2_l"
            cut_blocks_range_2_l_tmp = parse_cut_blocks(cut_transformer_blocks_2)
            cut_blocks_range_2_l = []
            for i, v in enumerate(cut_blocks_range_2_l_tmp):
                if v not in cut_blocks_range_l:
                    cut_blocks_range_2_l.append(v)
            cut_transformer_block_component_range(
                pretrained_name=pretrained_name,
                transformer=transformer,
                n_params_tx_orig=n_params_tx_orig,
                sorted_ret_dict=sorted_ret_dict,
                cut_blocks_l=cut_blocks_range_2_l,  # note, "cut_hybrid" has cut_blocks_range_2 differently from "cut_fg"
                cut_component_except_l=cut_component_except_l,
                cut_fg_dict=cut_fg_dict,
                target_memory_budget=target_memory_budget,
                debug=debug,
            )
        else:
            raise ValueError(f"Unknown cut_transformer_type={cut_transformer_type} is given.")

        # Check if all components within a transformer block is replaced, if so, put identity_block.
        cut_blocks_l = []
        if isinstance(transformer, SD3Transformer2DModel):
            for k, v in cut_fg_dict.items():
                if len(v) == 5 or "all" in v:
                    cut_blocks_l.append(k)
                    if not isinstance(transformer.transformer_blocks[k], Identity_Block_SD):
                        transformer.transformer_blocks[k] = Identity_Block_SD()
        elif isinstance(transformer, FluxTransformer2DModel):
            for k, v in cut_fg_dict.items():
                if k < transformer.config.num_layers:
                    if len(v) == 5 or "all" in v:
                        cut_blocks_l.append(k)
                        if not isinstance(transformer.transformer_blocks[k], Identity_Block_Flux):
                            transformer.transformer_blocks[k] = Identity_Block_Flux()
                else:
                    if len(v) == 3 or "all" in v:
                        cut_blocks_l.append(k)
                        if not isinstance(
                            transformer.single_transformer_blocks[
                                k - transformer.config.num_layers
                            ],
                            Identity_Block_Flux_Single,
                        ):
                            transformer.single_transformer_blocks[
                                k - transformer.config.num_layers
                            ] = Identity_Block_Flux_Single()

    return cut_blocks_l, cut_fg_dict


def load_and_cut_transformer(
    pretrained_name: str,
    TransformerClass: Union[SD3Transformer2DModel, FluxTransformer2DModel] = None,
    transformer_quant_type: str = None,
    cut_transformer_blocks: str = None,
    cut_transformer_blocks_2: str = None,
    cut_transformer_components_excluded: str = None,
    cut_transformer_type: str = "cut_blk_manual",
    target_memory_budget: float = None,
    benchmark_type: str = "hpsv2",
    metric_output_dir: str = None,
    weight_dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = None,
    debug: bool = True,
):
    """Load a pretrained transformer, apply pruning, and optionally quantize."""
    # 1. Get transformer Class
    if TransformerClass is None:
        TransformerClass, _ = get_transformer_and_pipeline_class(pretrained_name=pretrained_name)
    if debug:
        print(f"[INFO] Model: {pretrained_name}")

    # 2. Load a transformer model
    transformer = TransformerClass.from_pretrained(
        pretrained_name,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        cache_dir=cache_dir,
    )
    n_params_tx_orig = get_num_params(transformer)
    if debug:
        print(f"[INFO] Num Params of the ORIGINAL model {n_params_tx_orig/1000**3:0.3f} ")

    # 3. Cut the transformer model
    cut_blocks_l, cut_fg_dict = cut_transformer(
        pretrained_name=pretrained_name,
        transformer=transformer,
        cut_transformer_blocks=cut_transformer_blocks,
        cut_transformer_blocks_2=cut_transformer_blocks_2,
        cut_transformer_components_excluded=cut_transformer_components_excluded,
        cut_transformer_type=cut_transformer_type,
        target_memory_budget=target_memory_budget,
        benchmark_type=benchmark_type,
        metric_output_dir=metric_output_dir,
        debug=debug,
    )
    n_params_tx_cut = get_num_params(transformer)

    # 4. Quantize model
    if transformer_quant_type in ["bnb_nf4"]:
        quantize_linear_layers(transformer)

    return transformer, cut_blocks_l, cut_fg_dict, n_params_tx_orig, n_params_tx_cut


def load_and_cut_transformer_return_pipeline(
    pretrained_name: str,
    transformer_pretrained: str,
    transformer_quant_type: str = None,
    cut_transformer_blocks: str = None,
    cut_transformer_blocks_2: str = None,
    cut_transformer_components_excluded: str = None,
    cut_transformer_type: str = "cut_blk_manual",
    target_memory_budget: float = None,
    benchmark_type: str = "hpsv2",
    metric_output_dir: str = None,
    weight_dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = None,
    debug: bool = True,
):
    """Load a pruned transformer and assemble it into a full diffusers pipeline."""
    # 1. Get Pipeline Class
    TransformerClass, PipelineClass = get_transformer_and_pipeline_class(
        pretrained_name=pretrained_name
    )

    # if large models are loaded that have sharded checkpoints.
    if (
        "stabilityai/stable-diffusion-3-large" in pretrained_name
        or "stabilityai/stable-diffusion-3.5-large" in pretrained_name
        or "black-forest-labs/FLUX.1" in pretrained_name
    ):
        import accelerate

        from model.big_modeling import my_load_checkpoint_and_dispatch

        accelerate.load_checkpoint_and_dispatch = my_load_checkpoint_and_dispatch

    (
        transformer,
        cut_blocks_l,
        cut_fg_dict,
        n_params_tx_orig,
        n_params_tx_cut,
    ) = load_and_cut_transformer(
        pretrained_name=transformer_pretrained,
        TransformerClass=TransformerClass,
        transformer_quant_type=transformer_quant_type,
        cut_transformer_blocks=cut_transformer_blocks,
        cut_transformer_blocks_2=cut_transformer_blocks_2,
        cut_transformer_components_excluded=cut_transformer_components_excluded,
        cut_transformer_type=cut_transformer_type,
        target_memory_budget=target_memory_budget,
        benchmark_type=benchmark_type,
        metric_output_dir=metric_output_dir,
        cache_dir=cache_dir,
        debug=debug,
    )
    pipe = PipelineClass.from_pretrained(
        pretrained_name,
        transformer=transformer,
        torch_dtype=weight_dtype,
        cache_dir=cache_dir,
    )

    return pipe, cut_blocks_l, cut_fg_dict, n_params_tx_orig, n_params_tx_cut


def load_distilled_pipeline(
    pretrained_name: str,
    transformer_pretrained: str = None,
    transformer_quant_type: str = None,
    cut_transformer_blocks: str = None,
    cut_transformer_blocks_component: str = "all",
    weight_dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = None,
    debug: bool = True,
):
    """Load a full pipeline, prune the transformer, and load distilled weights."""
    TransformerClass, PipelineClass = get_transformer_and_pipeline_class(
        pretrained_name=pretrained_name
    )
    if debug:
        print(
            f"[INFO] Model: {pretrained_name}, transformer_pretrained: {transformer_pretrained}, quant_type: {transformer_quant_type}"
        )

    # # Connfigure Quantisation Scheme
    print(f"[INFO] Load Original Pipeline")
    pipe = PipelineClass.from_pretrained(
        pretrained_name,
        torch_dtype=weight_dtype,
        cache_dir=cache_dir,
    )
    n_params_tx_orig = get_num_params(pipe.transformer)
    if debug:
        print(f"[INFO] Num Params of the ORIGINAL model {n_params_tx_orig/1000**3:0.3f} B")

    # Load distilled transformer model
    cut_blocks_l = parse_cut_blocks(cut_transformer_blocks)
    if cut_blocks_l:
        if debug:
            print(f"[INFO] Test model with removed transformer blocks {cut_blocks_l}")
        n_params_tx_cut = cut_transformer_block_component(
            transformer=pipe.transformer,
            pretrained_name=pretrained_name,
            cut_blocks_l=cut_blocks_l,
            cut_transformer_blocks_component=cut_transformer_blocks_component,
        )

        if transformer_quant_type not in ["bnb_nf4"] and transformer_pretrained is not None:
            if "checkpoint" in transformer_pretrained:
                model_data_path = os.path.join(transformer_pretrained, "model.safetensors")
            else:
                model_data_path = os.path.join(
                    transformer_pretrained,
                    "transformer/diffusion_pytorch_model.safetensors",
                )
            model_data = safetensors.torch.load_file(model_data_path)
            pipe.transformer.load_state_dict(model_data, strict=False)

        if debug:
            print(
                f"[INFO] Num Params of the DISTILLED model {n_params_tx_cut/1000**3:0.3f} B ({n_params_tx_cut/n_params_tx_orig*100:0.3f}% of the original model)"
            )
    else:
        n_params_tx_cut = 0
        if debug:
            print(f"[INFO] Test original model without any cutting.")

    # Quantize model
    if transformer_quant_type in ["bnb_nf4"]:
        quantize_linear_layers(pipe.transformer)

    return pipe, cut_blocks_l, n_params_tx_orig, n_params_tx_cut


if __name__ == "__main__":
    # load images
    img_path = "assets/imgs/img_1k_1k.png"
    img = Image.open(img_path).convert("RGB")
    img = preprocess(ImageOps.exif_transpose(img))
    dtype = torch.float16

    # test sd3
    name = "sd3.5"
    vae = get_vae(name, "stabilityai/stable-diffusion-3.5-medium", dtype=dtype, device="cuda")
    print(f"Num of Params for {name}: {get_num_params(vae)}")
    latent = vae_encode(
        name=name,
        vae=vae,
        images=img.to(dtype=dtype, device="cuda"),
        sample_posterior=True,
    )
    samples = vae_decode(name, vae, latent)
    samples = (
        torch.clamp(127.5 * samples + 128.0, 0, 255)
        .permute(0, 2, 3, 1)
        .to("cpu", dtype=torch.uint8)
        .numpy()[0]
    )
    pred_image = Image.fromarray(samples, "RGB")
    pred_image.save(f"test_{name}.png")

    # test dc-ae
    name = "dc-ae"
    vae = get_vae(name, "mit-han-lab/dc-ae-f32c32-sana-1.0", dtype=dtype, device="cuda")
    print(f"Num of Params for {name}: {get_num_params(vae)}")
    latent = vae_encode(
        name=name,
        vae=vae,
        images=img.to(dtype=dtype, device="cuda"),
        sample_posterior=True,
    )
    samples = vae_decode(name, vae, latent)
    samples = (
        torch.clamp(127.5 * samples + 128.0, 0, 255)
        .permute(0, 2, 3, 1)
        .to("cpu", dtype=torch.uint8)
        .numpy()[0]
    )
    pred_image = Image.fromarray(samples, "RGB")
    pred_image.save(f"test_{name}.png")

    # test load_and_cut_transformer
    (
        transformer,
        cut_blocks_l,
        cut_fg_dict,
        n_params_tx_orig,
        n_params_tx_cut,
    ) = load_and_cut_transformer(
        pretrained_name="stabilityai/stable-diffusion-3.5-large-turbo",
        cut_transformer_blocks="25-29,31-35",
        cut_transformer_blocks_2="13-19",
        cut_transformer_components_excluded="attn",
        cut_transformer_type="cut_hybrid",
        target_memory_budget=0.71,
        benchmark_type="hpsv2",
        cache_dir="/scratch1/datasets/hf_models_cache/",
        metric_output_dir="assets/precomputed_metrics/hpsv2/stable-diffusion-3.5-large-turbo.json",
        debug=True,
    )
    print(f"[INFO] cut_blocks_l={cut_blocks_l}")
    print(f"[INFO] cut_fg_dict={cut_fg_dict}")
    print(f"[INFO] n_params_tx_cut={n_params_tx_cut}, n_params_tx_orig={n_params_tx_orig}")
