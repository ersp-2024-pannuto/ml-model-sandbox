#!/bin/bash

# Detect OS and set Python executable
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON_EXEC=python3.10
else
    PYTHON_EXEC=python
fi

# Define the full list of users
# 20 21 24 27 29 31 32 33 34 36
users=(3 5 6 7 10 12 13 18 20 21)

# Loop over each user as the test user
for test_user in "${users[@]}"; do
    rm -f ./datasets_wisdm/augmented_dataset.pkl
    rm -f ./datasets_wisdm/processed_dataset.pkl

    train_users=()
    for u in "${users[@]}"; do
        if [ "$u" != "$test_user" ]; then
            train_users+=("$u")
        fi
    done

    echo "Running LOO: test=$test_user, train=${train_users[*]}"

    # Build and run the command
    cmd="$PYTHON_EXEC train.py \
        --dataset-dir ./datasets_wisdm \
        --split-method 2 \
        --num-time-steps 200 \
        --sample-step 20 \
        --epochs 10 \
        --labels Jogging Walking Upstairs Standing Downstairs \
        --columns-to-keep acc_x acc_y acc_z \
        --num-features 3 \
        --model-type cnn \
        --train-users ${train_users[*]} \
        --test-users $test_user"

    echo "$cmd"
    eval "$cmd"
done