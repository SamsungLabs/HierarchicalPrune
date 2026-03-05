#!/bin/bash

OBJECT_DETECTOR_FOLDER="../assets/pretrained_models/geneval/"
MMDET_FOLDER="./mmdetection/"

# download object detector model
if [ ! -d "$OBJECT_DETECTOR_FOLDER" ]; then
    echo "Model path does not exist. Running download_models.sh..."
    mkdir -p "$OBJECT_DETECTOR_FOLDER"
    bash geneval/evaluation/download_models.sh $OBJECT_DETECTOR_FOLDER
fi

# (This will run once in the beginning) Set up MMDET
if [ ! -d "$MMDET_FOLDER" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection || exit ; git checkout 2.x
    pip install -v -e . --no-build-isolation
    cd ../
fi
