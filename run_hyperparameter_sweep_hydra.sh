#!/bin/bash
# Using Hydra multi-run for hyperparameter sweep (more elegant)

# This uses Hydra's multi-run feature to automatically run all combinations
# Each group will contain 3 runs (one per seed)

# Customize these values as needed:
MODEL="lstm_img"
INPUTS="img_only"
LEARNING_RATES="0.01,0.001,0.0001"
LAMBDA_REGS="0.01,0.001,0.0001"
SEEDS="0,1,2"

echo "Starting hyperparameter sweep with Hydra multi-run..."
echo "Model: ${MODEL}"
echo "Inputs: ${INPUTS}"
echo "Learning rates: ${LEARNING_RATES}"
echo "Lambda regs: ${LAMBDA_REGS}"
echo "Seeds: ${SEEDS}"
echo ""
echo "This will run all combinations (total: $(echo ${LEARNING_RATES} | tr ',' '\n' | wc -l) × $(echo ${LAMBDA_REGS} | tr ',' '\n' | wc -l) × $(echo ${SEEDS} | tr ',' '\n' | wc -l) = $(($(echo ${LEARNING_RATES} | tr ',' '\n' | wc -l) * $(echo ${LAMBDA_REGS} | tr ',' '\n' | wc -l) * $(echo ${SEEDS} | tr ',' '\n' | wc -l))) runs)"
echo ""

python train.py -m \
    model=${MODEL} \
    inputs=${INPUTS} \
    training.learning_rate=${LEARNING_RATES} \
    training.lambda_reg=${LAMBDA_REGS} \
    seed=${SEEDS}

# This will create groups like:
# - lstm_img_lr_1e-03_lambda_1e-02_scheduler_... (3 runs: seed 0, 1, 2)
# - lstm_img_lr_1e-03_lambda_1e-03_scheduler_... (3 runs: seed 0, 1, 2)
# - lstm_img_lr_1e-03_lambda_1e-04_scheduler_... (3 runs: seed 0, 1, 2)
# ... and so on for all combinations


