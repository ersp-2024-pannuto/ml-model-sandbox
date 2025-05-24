#!/bin/bash
# Detect OS and set Python executable
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON_EXEC=python3.10
else
    PYTHON_EXEC=python
fi

$PYTHON_EXEC train.py \
  --dataset-dir ./datasets_audio \
  --split-method 3 \
  --num-time-steps 100 \
  --sample-step 20 \
  --epochs 10 \
  --n-folds 10 \
  --labels zero one two three four five six seven eight nine \
  --model-type cnn \
  --augmentations 5