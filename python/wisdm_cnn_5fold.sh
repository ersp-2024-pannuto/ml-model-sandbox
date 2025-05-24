#!/bin/bash
# Detect OS and set Python executable
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON_EXEC=python3.10
else
    PYTHON_EXEC=python
fi

$PYTHON_EXEC train.py \
  --dataset-dir ./datasets_wisdm \
  --split-method 3 \
  --num-time-steps 200 \
  --sample-step 100 \
  --epochs 10 \
  --n-folds 5 \
  --labels Jogging Walking Upstairs Standing Downstairs \
  --model-type cnn \
  --columns-to-keep acc_x acc_y acc_z \
  --num-features 3 \
  --users 1 2 3 4 5 6 7 8 9 10