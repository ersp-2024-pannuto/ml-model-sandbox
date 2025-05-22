#!/bin/bash

# Define the full list of users
users=(1 8)

# Loop over each user as the test user
for test_user in "${users[@]}"; do
    rm ./datasets_audio/augmented_dataset.pkl
    rm ./datasets_audio/processed_dataset.pkl
    train_users=()
    for u in "${users[@]}"; do
        if [ "$u" != "$test_user" ]; then
            train_users+=("$u")
        fi
    done

    echo "Running LOO: test=$test_user, train=${train_users[*]}"

    # Run the training command
    python3.10 train.py \
        --dataset-dir ./datasets_audio \
        --split-method 2 \
        --num-time-steps 100 \
        --sample-step 20 \
        --epochs 10 \
        --labels zero one two three four five six seven eight nine \
        --train-users "${train_users[@]}" \
        --test-users "$test_user"
done