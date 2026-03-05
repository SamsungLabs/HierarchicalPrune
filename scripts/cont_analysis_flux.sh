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

    ##### FLUX-SCHNELL #####
    printf "/* --- FLUX-SCHNELL --- */\n"
    # Inference using a original transformer
    CUDA_VISIBLE_DEVICES=0 python cont_analysis.py \
        --transformer_pretrained="" \
        --pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell" \
        --benchmark_type="$i" \
        --benchmark_take="$TAKE" \
        --output_dir="outputs/results_cont_analysis/$i" \
        --cache_dir="PATH_TO_CACHE_DIR" \
        --seed=0 \
        --height=1024 \
        --width=1024 \
        --num_inference_steps=4 \
        --guidance_scale=0 \
        --max_sequence_length=256 \
        --mixed_precision=bf16 \
        --cut_transformer_blocks="" \
        --debug
done
