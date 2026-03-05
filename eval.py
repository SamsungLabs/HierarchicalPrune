"""Evaluation codebase for original/pruned/distilled diffusion models.

Generates images and computes benchmark scores (HPSv2, GenEval) to evaluate the quality of original/pruned/distilled transformer models.
Supports both quantised and non-quantised models.
"""

import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from model.builder import load_and_cut_transformer_return_pipeline, load_distilled_pipeline
from utils.utils import return_none


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--transformer_pretrained",
        type=return_none,
        default=None,
        required=False,
        help="Path to pretrained Transformer model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_quant_type",
        type=return_none,
        default=None,
        required=False,
        help="Quantisation type for transformer. ['bnb_nf4', etc.]",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
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
        "--benchmark_take",
        type=return_none,
        default=None,
        help="The number of images that are taken for evaluation. Default is None, which means all samples are used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--metric_output_dir",
        type=return_none,
        default=None,
        help="The output directory where the pre-computed metric results are stored.",
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
        "--max_sequence_length",
        type=int,
        default=256,
        help=("The maximum sequence length that the pipeline takes as an input for the prompt."),
    )
    parser.add_argument(
        "--validation_prompt",
        type=return_none,
        default=None,
        help="A prompt given to a model.",
    )
    parser.add_argument(
        "--validation_prompts_config",
        type=str,
        default="configs/val_prompts.yml",
        help="A list of prompts config path.",
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
            "none",
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

    # Load Pipeline
    # use case 1: loading original/quantised pipeline (no cutting, distillation applied)
    # use case 2: loading distilled transformer from a checkpoint file.
    if (args.transformer_pretrained is None and args.cut_tx_type == "none") or (
        args.transformer_pretrained is not None and "checkpoint-" in args.transformer_pretrained
    ):
        pipe, cut_blocks_l, n_params_tx_orig, n_params_tx_cut = load_distilled_pipeline(
            pretrained_name=args.pretrained_model_name_or_path,
            transformer_pretrained=args.transformer_pretrained,
            transformer_quant_type=args.transformer_quant_type,
            cut_transformer_blocks=args.cut_transformer_blocks,
            weight_dtype=weight_dtype,
            cache_dir=args.cache_dir,
            debug=args.debug,
        )
        cut_fg_dict = None
    else:
        (
            pipe,
            cut_blocks_l,
            cut_fg_dict,
            n_params_tx_orig,
            n_params_tx_cut,
        ) = load_and_cut_transformer_return_pipeline(
            pretrained_name=args.pretrained_model_name_or_path,
            transformer_pretrained=args.transformer_pretrained
            if args.transformer_pretrained is not None
            else args.pretrained_model_name_or_path,
            transformer_quant_type=args.transformer_quant_type,
            cut_transformer_blocks=args.cut_transformer_blocks,
            cut_transformer_blocks_2=args.cut_transformer_blocks_2,
            cut_transformer_components_excluded=args.cut_transformer_components_excluded,
            cut_transformer_type=args.cut_tx_type,
            target_memory_budget=float(args.target_memory_budget)
            if args.target_memory_budget is not None
            else None,
            benchmark_type=args.benchmark_type,
            metric_output_dir=args.metric_output_dir,
            weight_dtype=weight_dtype,
            cache_dir=args.cache_dir,
            debug=args.debug,
        )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Save Config
    take = int(args.benchmark_take) if args.benchmark_take is not None else None
    seed = np.random.randint(10000) if args.use_rand_seed else args.seed
    if args.transformer_pretrained is not None:
        if "checkpoint" in args.transformer_pretrained.split("/")[-1]:
            path_name = args.transformer_pretrained.split("/")[-2]
        else:
            path_name = args.transformer_pretrained.split("/")[-1]
    else:
        path_name = args.pretrained_model_name_or_path.split("/")[-1]
    full_output_dir_path = Path(args.output_dir).joinpath(path_name)
    config_gen_img = {
        "args": vars(args),
        "transformer_pretrained": args.transformer_pretrained,
        "transformer_quant_type": args.transformer_quant_type,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "full_output_dir_path": str(full_output_dir_path),
        "benchmark_type": args.benchmark_type,
        "benchmark_take": take,
        "height": height,
        "width": width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "cut_blocks_l": cut_blocks_l,
        "cut_blocks_l_2": args.cut_transformer_blocks_2,
        "cut_transformer_components_excluded": args.cut_transformer_components_excluded,
        "cut_tx_type": args.cut_tx_type,
        "target_memory_budget": args.target_memory_budget,
        "n_params_transformer_original": n_params_tx_orig,
        "n_params_transformer_distilled": n_params_tx_cut,
        "cut_fg_dict": cut_fg_dict,
        "env_seed": args.seed,
        "infer_seed": seed,
    }
    full_output_dir_path.mkdir(parents=True, exist_ok=True)
    if not full_output_dir_path.joinpath(
        "config.json"
    ).is_file():  # Let's not overwrite on exising config file
        with open(full_output_dir_path.joinpath("config.json"), "w") as f:
            json.dump(config_gen_img, f, indent=4)
        if args.debug:
            print(f"[INFO] Saving a config file to {full_output_dir_path.joinpath('config.json')}")

    done_already = False
    # Configure prompts and evaluation function according to the data type
    if args.benchmark_type.lower() in ["hps", "hpsv2"]:
        import metrics.HPSv2.hpsv2 as hpsv2
        from metrics.HPSv2.hpsv2.evaluation import evaluate

        #### Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo) ####
        all_prompts = hpsv2.benchmark_prompts("all")
        num_total_prompts = 0
        for style, prompts in all_prompts.items():
            num_total_prompts += len(prompts[:take])
        progress_bar = tqdm(
            range(0, num_total_prompts),
            desc=args.benchmark_type,
            dynamic_ncols=True,
        )

        #### Run Inferences over all prompts ####
        for style, prompts in all_prompts.items():
            # Make a directory before generating and saving images
            full_output_dir_path.joinpath(style).mkdir(parents=True, exist_ok=True)
            prompts = prompts[:take]
            num_prompts = len(prompts)
            num_skipped_prompts = 0

            for idx, prompt in enumerate(prompts):
                if full_output_dir_path.joinpath(style, f"{idx:05d}.png").is_file():
                    num_skipped_prompts += 1
                    continue

                seed_everything(args.seed, verbose=False)
                with torch.no_grad():
                    image = pipe(
                        prompt,
                        negative_prompt="",
                        height=height,
                        width=width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        max_sequence_length=args.max_sequence_length,
                        generator=torch.Generator("cpu").manual_seed(seed),
                    ).images[0]
                progress_bar.update(1)
                # Save Image
                image.save(full_output_dir_path.joinpath(style, f"{idx:05d}.png"))
                progress_bar.set_postfix_str(
                    f"Saving an image to {full_output_dir_path.joinpath(style, f'{idx:05d}.png')}"
                )
                # if args.debug:
                #     print(
                #         f"[INFO] Saving an image to {full_output_dir_path.joinpath(style, f'{idx:05d}.png')}"
                #     )
        #### Only when num_skipped_prompts == num_prompts
        #### Check if config contain the output results already
        if num_prompts == num_skipped_prompts:
            with open(full_output_dir_path.joinpath("config.json")) as f:
                results = json.load(f)
            if results.get(args.benchmark_type) is not None:
                print(
                    f"[INFO] Skpping {args.benchmark_type} metric evaluation for {full_output_dir_path} as it's done already"
                )
                done_already = True

        #### Run Evaluation on generated images ####
        if not done_already:
            data_path = os.path.join(hpsv2.root_path, "datasets/benchmark")
            score = evaluate(
                mode="benchmark",
                data_path=data_path,
                root_dir=str(full_output_dir_path.resolve()),
                checkpoint_path=None,
            )

            #### Aggregate HPSv2 Score Results ####
            config_gen_img[args.benchmark_type] = {}
            all_score = []
            for model_id, data in score.items():
                for style, res in data.items():
                    config_gen_img[args.benchmark_type][style] = {}

                    avg_score = [np.mean(res[i : i + 80]) for i in range(0, len(res), 80)]
                    all_score.extend(res)

                    config_gen_img[args.benchmark_type][style]["Avg"] = f"{np.mean(avg_score):.5f}"
                    config_gen_img[args.benchmark_type][style]["Std"] = f"{np.std(avg_score):.5f}"

            config_gen_img[args.benchmark_type]["Overall_Score"] = {}
            config_gen_img[args.benchmark_type]["Overall_Score"][
                "Avg"
            ] = f"{np.mean(all_score):.5f}"
            config_gen_img[args.benchmark_type]["Overall_Score"]["Std"] = f"{np.std(all_score):.5f}"

            with open(full_output_dir_path.joinpath("results.json"), "w") as f:
                json.dump(score, f, indent=4)

    elif args.benchmark_type.lower() in ["geneval"]:
        #### Get benchmark prompts
        metadata_path = os.path.join(f"metrics/geneval/prompts/evaluation_metadata.jsonl")
        NUM_IMAGES_PER_PROMPT = 4
        with open(metadata_path) as f:
            metadatas = [json.loads(line) for line in f]

        metadatas = metadatas[:take]
        num_prompts = len(metadatas)
        num_skipped_prompts = 0
        progress_bar = tqdm(
            range(0, len(metadatas)),
            desc=args.benchmark_type,
            dynamic_ncols=True,
        )
        for index, metadata in enumerate(metadatas):
            seed_everything(args.seed, verbose=False)

            outpath = full_output_dir_path.joinpath("imgs", f"{index:0>5}")
            outpath.mkdir(parents=True, exist_ok=True)

            prompt = metadata["prompt"]
            # print(f"[INFO] Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

            sample_path = outpath.joinpath("samples")
            sample_path.mkdir(parents=True, exist_ok=True)

            with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
                json.dump(metadata, f)

            sample_count = 0
            if sample_path.joinpath(f"{sample_count:05d}.png").is_file():
                num_skipped_prompts += 1
                continue

            with torch.no_grad():
                images = pipe(
                    prompt,
                    negative_prompt="",
                    height=height,
                    width=width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                ).images
            progress_bar.update(1)
            # Save Image
            for img in images:
                img.save(sample_path.joinpath(f"{sample_count:05d}.png"))
                sample_count += 1
            progress_bar.set_postfix_str(
                f"Saving {NUM_IMAGES_PER_PROMPT} images to {sample_path.joinpath(f'{sample_count:05d}.png')}"
            )
            # if args.debug:
            #     print(
            #         f"[INFO] Saving {NUM_IMAGES_PER_PROMPT} images to {sample_path.joinpath(f'{sample_count:05d}.png')}"
            #     )
        #### Only when num_skipped_prompts == num_prompts
        #### Check if config contain the output results already
        if num_prompts == num_skipped_prompts:
            with open(full_output_dir_path.joinpath("config.json")) as f:
                results = json.load(f)
            if results.get(args.benchmark_type) is not None:
                print(
                    f"[INFO] Skpping {args.benchmark_type} metric evaluation for {full_output_dir_path} as it's done already"
                )
                done_already = True

        #### Run Evaluation on generated images ####
        if not done_already:
            target_py = str(
                Path(__file__)
                .parent.joinpath("metrics/geneval/evaluation/evaluate_images.py")
                .resolve()
            )
            from utils.utils import run_python

            img_folder = full_output_dir_path.joinpath("imgs")
            model_path = "assets/pretrained_models/geneval"
            run_python(
                target_py,
                f"{str(img_folder.resolve())} --outfile {str(full_output_dir_path.joinpath('results.jsonl').resolve())} --model-path {model_path}",
            )

            #### Aggregate GenEval Score Results ####
            # Load results
            df = pd.read_json(
                str(full_output_dir_path.joinpath("results.jsonl").resolve()),
                orient="records",
                lines=True,
            )

            # Measure overall success
            print("GenEval Summary")
            print("=======")
            print(f"[INFO] Total images: {len(df)}")
            print(f"[INFO] Total prompts: {len(df.groupby('metadata'))}")
            print(f"[INFO] % correct images: {df['correct'].mean():.2%}")
            print(f"[INFO] % correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
            print()

            # By group
            task_scores = []
            print("Task breakdown")
            print("==============")
            config_gen_img[args.benchmark_type] = {}
            for tag, task_df in df.groupby("tag", sort=False):
                task_scores.append(task_df["correct"].mean())
                print(
                    f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})"
                )
                config_gen_img[args.benchmark_type][
                    tag
                ] = f"{task_df['correct'].mean():.3%} ({task_df['correct'].sum()} / {len(task_df)})"
            print()

            print(f"[INFO] Overall score (avg. over tasks): {np.mean(task_scores):.5f}")
            config_gen_img[args.benchmark_type]["Overall_Score"] = f"{np.mean(task_scores):.5f}"

    elif args.benchmark_type.lower() in ["image_reward"]:
        import ImageReward as RM

        #### Get benchmark prompts
        metadata_path = f"metrics/image_reward/benchmark-prompts.json"
        NUM_IMAGES_PER_PROMPT = 10
        PIPE_BATCH_SIZE = 2
        with open(metadata_path) as f:
            prompt_with_id_list = json.load(f)

        # generate output image and result folders
        full_output_dir_path.joinpath("imgs").mkdir(parents=True, exist_ok=True)

        prompt_with_id_list = prompt_with_id_list[:take]
        num_prompts = len(prompt_with_id_list)
        num_skipped_prompts = 0
        progress_bar = tqdm(
            range(0, len(prompt_with_id_list)),
            desc=args.benchmark_type,
            dynamic_ncols=True,
        )
        for index, prompt_id in enumerate(prompt_with_id_list):
            seed_everything(args.seed, verbose=False)

            id = prompt_id["id"]
            prompt = prompt_id["prompt"]
            # print(f"[INFO] Prompt ({index: >3}/{len(prompt_with_id_list)}): '{id}' '{prompt}'")

            if full_output_dir_path.joinpath(
                "imgs", f"{id}_{NUM_IMAGES_PER_PROMPT-1}.png"
            ).is_file():
                num_skipped_prompts += 1
                continue

            sample_count = 0
            for i in range(PIPE_BATCH_SIZE):
                with torch.no_grad():
                    images = pipe(
                        prompt,
                        negative_prompt="",
                        height=height,
                        width=width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        max_sequence_length=args.max_sequence_length,
                        generator=torch.Generator("cpu").manual_seed(seed + i),
                        num_images_per_prompt=NUM_IMAGES_PER_PROMPT // PIPE_BATCH_SIZE,
                    ).images
                # Save Image
                for img in images:
                    img.save(full_output_dir_path.joinpath("imgs", f"{id}_{sample_count}.png"))
                    sample_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"Saving {NUM_IMAGES_PER_PROMPT} images to {full_output_dir_path.joinpath('imgs', f'{id}_{sample_count}.png')}"
            )
            # if args.debug:
            #     print(
            #         f"[INFO] Saving {NUM_IMAGES_PER_PROMPT} images to {full_output_dir_path.joinpath('imgs', f'{id}_{sample_count}.png')}"
            #     )
        #### Only when num_skipped_prompts == num_prompts
        #### Check if config contain the output results already
        if num_prompts == num_skipped_prompts:
            with open(full_output_dir_path.joinpath("config.json")) as f:
                results = json.load(f)
            if results.get(args.benchmark_type) is not None:
                print(
                    f"[INFO] Skpping {args.benchmark_type} metric evaluation for {full_output_dir_path} as it's done already"
                )
                done_already = True

        #### Run Evaluation on generated images ####
        if not done_already:
            # Load the ImageReward model
            model = RM.load(name="ImageReward-v1.0")

            count = 0
            all_scores = []
            img_score = {}
            for prompt_id in tqdm(
                prompt_with_id_list, desc=f"ImageReward {NUM_IMAGES_PER_PROMPT} images / prompt"
            ):
                id = prompt_id["id"]
                prompt = prompt_id["prompt"]
                for i in range(NUM_IMAGES_PER_PROMPT):
                    img_path = full_output_dir_path.joinpath("imgs", f"{id}_{i}.png")
                    if img_path.is_file():
                        with torch.no_grad():
                            score = model.score(prompt, str(img_path.resolve()))
                        all_scores.append(score)
                        count += 1
                        img_score[img_path.name] = score

            #### Aggregate ImageReward Score Results ####
            image_reward_value = sum(all_scores) / count
            print(f"[INFO] Image Reward {image_reward_value}")
            if count < len(prompt_with_id_list) * NUM_IMAGES_PER_PROMPT:
                print(
                    f"[INFO][WARNING] Image Generation has not been fully complete yet {count}/{len(prompt_with_id_list) * NUM_IMAGES_PER_PROMPT}"
                )

            config_gen_img[args.benchmark_type] = {}
            config_gen_img[args.benchmark_type]["Avg"] = f"{np.mean(all_scores):.5f}"
            config_gen_img[args.benchmark_type]["Std"] = f"{np.std(all_scores):.5f}"

            with open(full_output_dir_path.joinpath("results.json"), "w") as f:
                json.dump(img_score, f, indent=4)
    else:
        raise ValueError(f"Invalid argument: args.benchmark_type=={args.benchmark_type} is given.")

    ##### Write Results #####
    if not done_already:
        with open(full_output_dir_path.joinpath("config.json"), "w") as f:
            json.dump(config_gen_img, f, indent=4)

    ##### Reset Memory #####
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    args = parse_args()
    main(args)
