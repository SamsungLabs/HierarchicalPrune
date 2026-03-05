#/bin/bash

OBJECT_DETECTOR_FOLDER="../assets/pretrained_models/geneval/"

IMAGE_FOLDER="tmp_geneval/imgs"
RESULTS_FOLDER="tmp_geneval/results"
Generate_Image=true

# (optional) run Image Generation
if [ "$Generate_Image" = true ]; then
    # mkdir tmp_geneval
    mkdir -p $IMAGE_FOLDER

    python geneval/generation/diffusers_generate.py \
        "geneval/prompts/evaluation_metadata.jsonl" \
        --model "runwayml/stable-diffusion-v1-5" \
        --outdir "$IMAGE_FOLDER"
fi

# GenEval's Evaluation
if [ ! -d "$RESULTS_FOLDER" ]; then
    mkdir -p $RESULTS_FOLDER
fi

python geneval/evaluation/evaluate_images.py \
    "$IMAGE_FOLDER" \
    --outfile "$RESULTS_FOLDER/results.jsonl" \
    --model-path "$OBJECT_DETECTOR_FOLDER"

python geneval/evaluation/summary_scores.py \
    "$RESULTS_FOLDER/results.jsonl"
