#!/bin/bash

## declare an array variable
# Note that in order to run the evaluation based on the GenEval metric, you need to install GenEval.
# Please refer to Additional Setup in README.md for more details.
declare -a arr=("hpsv2")

## now loop through the above array
for i in "${arr[@]}"
do
    echo "/* -------- Running $i -------- */"
    printf "\n"

    # Update "transformer_pretrained", "transformer_quant_type", "cache_dir", "cut_transformer_blocks", "cut_tx_type"

    ##### FLUX-SCHNELL #####
    printf "/* --- FLUX-SCHNELL --- */\n"

    # Inference using a original transformer
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --transformer_pretrained="" \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell" \
        --benchmark_type="$i" \
        --output_dir="outputs/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=4 \
        --guidance_scale=0 \
        --max_sequence_length=256 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="" \
        --cut_tx_type="none" \
        --debug

    # Inference using a distilled transformer
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell" \
        --benchmark_type="$i" \
        --output_dir="outputs/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=4 \
        --guidance_scale=0 \
        --max_sequence_length=256 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="PRUNED_BLOCK_INDICES" \
        --cut_tx_type="cut_blk_manual" \
        --debug

    # Inference using a distilled transformer with W4A16 quantisation
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
        --transformer_quant_type="bnb_nf4" \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell" \
        --benchmark_type="$i" \
        --output_dir="outputs/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=4 \
        --guidance_scale=0 \
        --max_sequence_length=256 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="PRUNED_BLOCK_INDICES" \
        --cut_tx_type="cut_blk_manual" \
        --debug


    ##### FLUX-DEV #####
    printf "/* --- FLUX-DEV --- */\n"

    # Inference using a original transformer
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --transformer_pretrained="" \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
        --benchmark_type="$i" \
        --output_dir="outputs/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=50 \
        --guidance_scale=3.5 \
        --max_sequence_length=512 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="" \
        --cut_tx_type="none" \
        --debug

    # Inference using a distilled transformer
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
        --benchmark_type="$i" \
        --output_dir="outputs/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=50 \
        --guidance_scale=3.5 \
        --max_sequence_length=512 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="PRUNED_BLOCK_INDICES" \
        --cut_tx_type="cut_blk_manual" \
        --debug

    # Inference using a distilled transformer with W4A16 quantisation
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
        --transformer_quant_type="bnb_nf4" \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
        --benchmark_type="$i" \
        --output_dir="outputs/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=4 \
        --guidance_scale=0 \
        --max_sequence_length=512 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="PRUNED_BLOCK_INDICES" \
        --cut_tx_type="cut_blk_manual" \
        --debug
done
