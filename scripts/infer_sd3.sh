#!/bin/bash

# Update "transformer_pretrained", "transformer_quant_type", "cache_dir", "cut_transformer_blocks", "cut_tx_type"

##### SD3.5-Large Turbo #####

# Inference using a original transformer
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large-turbo" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=4 \
    --guidance_scale=0.0 \
    --max_sequence_length=512 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="" \
    --cut_tx_type="none" \
    --debug

# Inference using a distilled transformer
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large-turbo" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=4 \
    --guidance_scale=0.0 \
    --max_sequence_length=512 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="25-29,31-35" \
    --cut_tx_type="cut_blk_manual" \
    --debug

# Inference using a distilled transformer with W4A16 quantisation
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
    --transformer_quant_type="bnb_nf4" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large-turbo" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR/" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=4 \
    --guidance_scale=0.0 \
    --max_sequence_length=512 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="25-29,31-35" \
    --cut_tx_type="cut_blk_manual" \
    --debug



##### SD3.5-Medium #####

# Inference using a original transformer
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=28 \
    --guidance_scale=7.0 \
    --max_sequence_length=256 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="" \
    --cut_tx_type="none" \
    --debug

# Inference using a distilled transformer
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=28 \
    --guidance_scale=7.0 \
    --max_sequence_length=256 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="13,15,16,18,20,21" \
    --cut_tx_type="cut_blk_manual" \
    --debug

# Inference using a distilled transformer with W4A16 quantisation
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
    --transformer_quant_type="bnb_nf4" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR/" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=28 \
    --guidance_scale=7.0 \
    --max_sequence_length=256 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="13,15,16,18,20,21" \
    --cut_tx_type="cut_blk_manual" \
    --debug



##### SD3.5-Large #####

# Inference using a original transformer
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=28 \
    --guidance_scale=3.5 \
    --max_sequence_length=512 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="" \
    --cut_tx_type="none" \
    --debug

# Inference using a distilled transformer
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=28 \
    --guidance_scale=3.5 \
    --max_sequence_length=512 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="PRUNED_BLOCK_INDICES" \
    --cut_tx_type="cut_blk_manual" \
    --debug

# Inference using a distilled transformer with W4A16 quantisation
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --transformer_pretrained="PATH_TO_PRETRAINED_MODEL" \
    --transformer_quant_type="bnb_nf4" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-large" \
    --output_dir="outputs" \
    --cache_dir="PATH_TO_CACHE_DIR/" \
    --seed=0 \
    --height=1024 \
    --width=1024 \
    --num_inference_steps=28 \
    --guidance_scale=3.5 \
    --max_sequence_length=512 \
    --validation_prompt="" \
    --validation_prompts_config="configs/val_prompts.yml" \
    --mixed_precision=bf16 \
    --cut_transformer_blocks="PRUNED_BLOCK_INDICES" \
    --cut_tx_type="cut_blk_manual" \
    --debug
