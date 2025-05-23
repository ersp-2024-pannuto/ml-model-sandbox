#!/bin/bash

# Detect OS and set Python executable
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON_EXEC=python3.10
else
    PYTHON_EXEC=python
fi

# Define the full list of users
users=(1 8 9)

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

    # Run the training command
    $PYTHON_EXEC train.py \
        --dataset-dir ./datasets_audio \
        --split-method 2 \
        --num-time-steps 100 \
        --sample-step 20 \
        --epochs 10 \
        --labels zero one two three four five six seven eight nine \
        --train-users "${train_users[@]}" \
        --test-users "$test_user"
done