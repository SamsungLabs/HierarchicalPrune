#!/bin/bash

## declare an array variable
# Note that in order to run the evaluation based on the GenEval metric, you need to install GenEval.
# Please refer to Additional Setup in README.md for more details.
declare -a arr=("hpsv2")
TAKE=100

## now loop through the above array
for i in "${arr[@]}"
do
    echo "/* -------- Running $i -------- */"
    printf "\n"

    # Update "transformer_pretrained", "transformer_quant_type", "cache_dir"

    ##### SD3.5-Medium #####
    printf "/* --- SD3.5-Medium --- */\n"
    # Inference using a original transformer
    CUDA_VISIBLE_DEVICES=0 python cont_analysis.py \
        --transformer_pretrained="" \
        --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" \
        --benchmark_type="$i" \
        --benchmark_take="$TAKE" \
        --output_dir="outputs/results_cont_analysis/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=28 \
        --guidance_scale=7.0 \
        --max_sequence_length=256 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="" \
        --debug

    ##### SD3.5-Large #####
    printf "/* --- SD3.5-Large --- */\n"
    # Inference using a original transformer
    CUDA_VISIBLE_DEVICES=0 python cont_analysis.py \
        --transformer_pretrained="" \
        --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large" \
        --benchmark_type="$i" \
        --benchmark_take="$TAKE" \
        --output_dir="outputs/results_cont_analysis/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=28 \
        --guidance_scale=3.5 \
        --max_sequence_length=512 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="" \
        --debug

    ##### FLUX-DEV #####
    printf "/* --- FLUX-DEV --- */\n"
    # Inference using a original transformer
    CUDA_VISIBLE_DEVICES=0 python cont_analysis.py \
        --transformer_pretrained="" \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
        --benchmark_type="$i" \
        --benchmark_take="$TAKE" \
        --output_dir="outputs/results_cont_analysis/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=50 \
        --guidance_scale=3.5 \
        --max_sequence_length=512 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="" \
        --debug
done
