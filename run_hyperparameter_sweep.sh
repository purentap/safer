#!/bin/bash
# Script to run hyperparameter sweep with 3 seeds for each hyperparameter combination

# Define hyperparameter values
LEARNING_RATES=(0.001 0.0001 0.00001)
LAMBDA_REGS=(0.01 0.001 0.0001)
SEEDS=(0 1 2)

# Model and input configuration
MODEL="lstm_img"
INPUTS="img_only"

echo "Starting hyperparameter sweep..."
echo "Learning rates: ${LEARNING_RATES[@]}"
echo "Lambda regs: ${LAMBDA_REGS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""

# Loop through all combinations
for lr in "${LEARNING_RATES[@]}"; do
    for lambda_reg in "${LAMBDA_REGS[@]}"; do
        echo "Running: lr=${lr}, lambda_reg=${lambda_reg}"
        
        # Run 3 seeds for this hyperparameter combination
        for seed in "${SEEDS[@]}"; do
            echo "  Seed: ${seed}"
            python train.py \
                model=${MODEL} \
                inputs=${INPUTS} \
                training.learning_rate=${lr} \
                training.lambda_reg=${lambda_reg} \
                seed=${seed}
        done
        echo ""
    done
done

echo "All experiments completed!"

