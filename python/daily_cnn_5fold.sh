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
  --num-time-steps 200 \
  --sample-step 100 \
  --epochs 10 \
  --n-folds 5 \
  --labels standing_still walking_forward running_forward climb_up climb_down \
  --model-type cnn \
  --columns-to-keep acc_x acc_y acc_z \
  --num-features 3