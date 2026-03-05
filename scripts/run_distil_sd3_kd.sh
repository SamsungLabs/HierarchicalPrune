#!/bin/bash

### MODEL SELECTION & DATA, Cached MODEL PATH CONFIG ###
VAE_NAME="sd3.5"
VAE_PRETRAINED="stabilityai/stable-diffusion-3.5-medium"
MODEL_PRETRAINED="stabilityai/stable-diffusion-3.5-medium"
TRAIN_DATA_DIR="PATH_TO_TRAIN_DATA_DIR" # please adjust it if needed
MODEL_CACHE_DIR="PATH_TO_CACHE_DIR" # please adjust it if needed
METRIC_OUTPUT_DIR="assets/precomputed_metrics/hpsv2/stable-diffusion-3.5-medium.json"

### Training-related user arguments ###
RESOLUTION=1024
NUM_GPUS=8
BATCH_SIZE=16 # 4 when "SA" is used for KD_feat_type
GRAD_ACCUMULATION=1
NUM_DATALOADER_WORKERS=4
MAX_TRAIN_STEPS=64000
LR=1e-4
CHECKPOINTING_STEPS=12000

### Distillation-related user arguments ###
CUT_BLOCKS="13,15,16,18,20,21" # Transformer blocks that are removed or pruned.
CUT_COMP_EXCEPT="" # Transformer components that are NOT removed nor pruned.
CUT_RATIOS="1,1,1,1,1,1" # cut X * 100%
KD_FEAT_TYPE="LFImg,LFCond" # default is simply transformer output distillation. Another example is feature distillation: "SA,CA,LFImg,LFCond"
# option: ["", "SA,CA,LFImg,LFCond", "SA,CA", "SA,LFImg", "SA,LFCond", etc.]
# Basically, one can mix those 4 options ("SA", "CA", "LFImg", "LFCond") for those transformer layers that are not cut
KD_LOSS_SCALING=True
LAMBDA_TASK=1.0
LAMBDA_KD_OUT=1.0
LAMBDA_KD_FEAT=1e-4 # loss calibration is applied.
CUT_TX_TYPE="cut_blk_manual"

### Validation-related user arguments ###
VALIDATION_STEPS=$((CHECKPOINTING_STEPS/2))
NUM_INFER_STEPS=28
GUIDANCE_SCALE=7.0
NUM_VAL_IMAGES_PER_PROMPT=1
SAVE_VAL_IMAGES_OR_NOT=1
VAL_PROMPT_CONFIG="configs/val_prompts.yml"

CUT_BLOCKS_2=""
SCALING_RANGE=""

DATE=$(date "+%Y-%m-%d_%H:%M:%S")
OUTPUT_DIR="../distil_Diffusers_ckpts/outputs/"$DATE"_sd35-medium-distil"$KD_FEAT_TYPE"_"$CUT_TX_TYPE"_scaleRange"$SCALING_RANGE"_no_kd_task_lambda_kd_feat"$LAMBDA_KD_FEAT"_kd_loss_scaling"$KD_LOSS_SCALING"_res"$RESOLUTION"_bcz$((BATCH_SIZE*NUM_GPUS*GRAD_ACCUMULATION))_lr"$LR"_cut"$CUT_BLOCKS"_cut2"$CUT_BLOCKS_2"_rate"$CUT_RATIOS"_TrStep"$MAX_TRAIN_STEPS

StartTime=$(date +%s)
accelerate launch --config_file=configs/deepspeed_config.yaml --main_process_port=26503 \
    --num_processes=$NUM_GPUS --gradient_accumulation_steps=$GRAD_ACCUMULATION distil_sd3.py \
    --seed="12345" \
    --vae_name=$VAE_NAME \
    --vae_pretrained=$VAE_PRETRAINED \
    --pretrained_model_name_or_path=$MODEL_PRETRAINED \
    --metric_output_dir=$METRIC_OUTPUT_DIR --cut_tx_type=$CUT_TX_TYPE \
    --kd_loss_scaling_range=$SCALING_RANGE \
    --resolution=$RESOLUTION --torch_compile \
    --train_batch_size=$BATCH_SIZE --dataloader_num_workers=$NUM_DATALOADER_WORKERS \
    --gradient_accumulation_steps=$GRAD_ACCUMULATION --gradient_checkpointing \
    --max_train_steps=$MAX_TRAIN_STEPS --max_sequence_length="256" \
    --learning_rate=$LR --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --checkpointing_steps=$CHECKPOINTING_STEPS \
    --caption_type="both" \
    --instance_data_dir=$TRAIN_DATA_DIR \
    --cache_dir=$MODEL_CACHE_DIR \
    --output_dir=$OUTPUT_DIR \
    --cut_transformer_blocks=$CUT_BLOCKS --cut_transformer_blocks_2=$CUT_BLOCKS_2 --cut_transformer_components_excluded=$CUT_COMP_EXCEPT \
    --cut_transformer_blocks_ratios=$CUT_RATIOS \
    --kd_feat_type=$KD_FEAT_TYPE \
    --kd_loss_scaling=$KD_LOSS_SCALING \
    --lambda_task=$LAMBDA_TASK --lambda_kd_out=$LAMBDA_KD_OUT --lambda_kd_feat=$LAMBDA_KD_FEAT \
    --validation_steps=$VALIDATION_STEPS --num_inference_steps=$NUM_INFER_STEPS --guidance_scale=$GUIDANCE_SCALE \
    --num_validation_images=$NUM_VAL_IMAGES_PER_PROMPT --save_validation_images=$SAVE_VAL_IMAGES_OR_NOT \
    --validation_prompts_config=$VAL_PROMPT_CONFIG
EndTime=$(date +%s)
echo "** Training takes $((EndTime - StartTime)) seconds."
