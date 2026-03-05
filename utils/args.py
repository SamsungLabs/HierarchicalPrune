"""Shared argument definitions for SD3/Flux distillation training codebases ('distil_sd3.py', 'distil_flux.py')."""

import argparse
import os

from utils.utils import return_none


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--distill_type",
        type=str,
        default="latent",
        choices=[
            "latent",
            "pixel",
        ],
        required=False,
        help="Type of distillation. Options: ['latent', 'pixel']",
    )
    parser.add_argument(
        "--vae_name",
        type=str,
        default="sd3.5",
        choices=[
            "sd3",
            "sd3.5",
            "dc-ae",
            "flux",
        ],
        required=False,
        help="VAE name.",
    )
    parser.add_argument(
        "--vae_pretrained",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        choices=[
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3.5-medium",
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-large-turbo",
            "mit-han-lab/dc-ae-f32c32-sana-1.0",
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev",
        ],
        required=False,
        help="Path to pretrained VAE model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        choices=[
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3.5-medium",
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-large-turbo",
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev",
        ],
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
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--data_df_path",
        type=str,
        default=None,
        help=("Path to the parquet file serialized with compute_embeddings.py."),
    )
    parser.add_argument(
        "--cache_dir",
        type=return_none,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="[DEPRECATED] A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_prompts_config",
        type=str,
        default="configs/val_prompts.yml",
        help="A list of prompts config path used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--save_validation_images",
        type=bool,
        default=False,
        help="If set, save the validation images during training.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "[DEPRECATED] no longer used in this codebase any longer."
            "Run validation every X epochs. validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--metric_output_dir",
        type=return_none,
        default=None,
        help="The output directory where the pre-computed metric results are stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--torch_compile",
        default=False,
        action="store_true",
        help=(
            "Whether to use torch.compile() to the models. If set, torch.compile() is applied to all the models."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run validation every X steps. validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--transformer_precision",
        type=str,
        default=None,
        choices=["int4", "int8", "fp"],
        help=("Whether to use int4, int8, or normal fp precision for Transformer."),
    )
    parser.add_argument(
        "--t5_precision",
        type=str,
        default=None,
        choices=["int4", "int8", "fp"],
        help=("Whether to use int4, int8, or normal fp precision for T5 text encoder."),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
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
        default="cut_blk_manual",
        choices=[
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
        "--transformer_patch_size",
        type=int,
        default=1,
        help=("patch size of the transformer."),
    )
    parser.add_argument(
        "--transformer_init_type",
        type=str,
        default="partial_repeat",
        choices=["all_rand", "partial_zeros", "partial_rand", "partial_repeat"],
        required=False,
        help=("init type of the transformer."),
    )
    parser.add_argument(
        "--cut_transformer_blocks_ratios",
        type=return_none,
        default=1.0,
        help=(
            "Ratios of removed transformer blocks. usage example:"
            "(1) when specifying entire cutting ratio, use 1"
            "(2) when giving cutting ratio for each transformer blocks to be cut, use 1,1,0.5,0.5"
        ),
    )
    parser.add_argument(
        "--kd_feat_type",
        type=return_none,
        default=None,
        help=(
            "For the remaining transformer blocks that are not cut, we do feature distillation."
            "Use example are as follows:"
            "(1) when distilling Self Attention only, use SA"
            "(1.2) when distilling Self Attention 2 only, use SA2"
            "(2) when distilling Cross Attention only, use CA"
            "(3) when distilling Last Feature for image only, use LFImg"
            "(4) when distilling Last Feature for conditioning only, use LFCond"
            "(5) when distilling with mixture of different KD_feat types, use SA,SA2,CA,LFImg,LFCond"
        ),
    )
    parser.add_argument(
        "--kd_loss_scaling",
        type=return_none,
        default=None,
        help=(
            "For the distillation loss, we perform scaling."
            "Also, we freeze all the layer update for the blocks before the 1st removed block,"
            "which will reduce the memory requirements as well as speed up training."
        ),
    )
    parser.add_argument(
        "--kd_loss_scaling_range",
        type=return_none,
        default=None,
        help=(
            "For range of loss scaling. usage example:"
            "(1) when specifying individual block index, use 10,20"
            "(2) when giving a range of indices, use 10-20"
            "(3) when giving a list of ranges of indices, use 8-12,16-20"
        ),
    )
    parser.add_argument(
        "--lambda_task",
        type=float,
        default=0.0,
        help=("Weight of diffusion task loss to the total loss computation."),
    )
    parser.add_argument(
        "--lambda_kd_out",
        type=float,
        default=1.0,
        help=("Weight of KD-output loss to the total loss computation."),
    )
    parser.add_argument(
        "--lambda_kd_feat",
        type=float,
        default=0.0,
        help=("Weight of KD-feature loss to the total loss computation."),
    )
    parser.add_argument(
        "--caption_type",
        type=str,
        default="both",
        required=False,
        help=("caption type of the ye pop dataset."),
    )
    parser.add_argument(
        "--webdataset_url",
        type=str,
        default=None,
        help=(
            "[DEPRECATED] The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
