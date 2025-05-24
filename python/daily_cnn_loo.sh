#!/bin/bash

# Detect OS and set Python executable
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON_EXEC=python3.10
else
    PYTHON_EXEC=python
fi

# Define the full list of users
# 1 2 4 5 6 7 10 12
users=(1 10)

# Loop over each user as the test user
for test_user in "${users[@]}"; do
    rm -f ./datasets_audio/augmented_dataset.pkl
    rm -f ./datasets_audio/processed_dataset.pkl

    train_users=()
    for u in "${users[@]}"; do
        if [ "$u" != "$test_user" ]; then
            train_users+=("$u")
        fi
    done

    echo "Running LOO: test=$test_user, train=${train_users[*]}"

    # Build and run the command
    cmd="$PYTHON_EXEC train.py \
        --dataset-dir ./datasets_audio \
        --split-method 2 \
        --num-time-steps 200 \
        --sample-step 20 \
        --epochs 10 \
        --labels standing_still walking_forward running_forward climb_up climb_down \
        --columns-to-keep acc_x acc_y acc_z \
        --num-features 3 \
        --model-type cnn \
        --augmentations 0 \
        --train-users ${train_users[*]} \
        --test-users $test_user"

    echo "$cmd"
    eval "$cmd"
done