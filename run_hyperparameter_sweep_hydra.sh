#!/bin/bash
# Using Hydra multi-run for hyperparameter sweep (more elegant)

# This uses Hydra's multi-run feature to automatically run all combinations
# Each group will contain 3 runs (one per seed)

# Customize these values as needed:
MODEL="lstm_img"
#INPUTS="img_only"
#LEARNING_RATES="0.01,0.001,0.0001"
LAMBDA_REGS="0.1,0.01,0.001"
SEEDS="0,1,2"
EPOCHS="300"
PROJECT_NAME="img_only_cosine_annealing_sweep"
POLICY="OPENVLA"




<<'COMMENT'
# # Configuration 1: LR 0.01 -> 0.0001
# echo "Running Configuration 3: LR 0.01 -> 0.0001"
# python train.py -m \
#     model=${MODEL} \
#     training.use_scheduler=true \
#     training.learning_rate=0.01 \
#     scheduler.min_lr=0.0001 \
#     scheduler.factor=0.8 \
#     scheduler.patience=10 \
#     training.lambda_reg=${LAMBDA_REGS} \
#     training.n_epochs=${EPOCHS} \
#     wandb.project=${PROJECT_NAME} \
#     seed=${SEEDS}
# '''
# '''
# echo "Starting hyperparameter sweep with Hydra multi-run..."
# echo "Model: ${MODEL}"
# echo "Learning rates: ${LEARNING_RATES}"
# echo "Lambda regs: ${LAMBDA_REGS}"
# echo "Seeds: ${SEEDS}"
# echo ""
# echo "This will run all combinations (total: $(echo ${LEARNING_RATES} | tr ',' '\n' | wc -l) × $(echo ${LAMBDA_REGS} | tr ',' '\n' | wc -l) × $(echo ${SEEDS} | tr ',' '\n' | wc -l) = $(($(echo ${LEARNING_RATES} | tr ',' '\n' | wc -l) * $(echo ${LAMBDA_REGS} | tr ',' '\n' | wc -l) * $(echo ${SEEDS} | tr ',' '\n' | wc -l))) runs)"
# echo ""

# # Use conda run to execute in the vla-safe environment
# python train.py -m \
#     model=${MODEL} \
#     training.learning_rate=${LEARNING_RATES} \
#     training.lambda_reg=${LAMBDA_REGS} \
#     training.use_scheduler=false \
#     seed=${SEEDS}
# '''
# # This will create groups like:
# # - lstm_img_lr_1e-03_lambda_1e-02_scheduler_... (3 runs: seed 0, 1, 2)
# # - lstm_img_lr_1e-03_lambda_1e-03_scheduler_... (3 runs: seed 0, 1, 2)
# # - lstm_img_lr_1e-03_lambda_1e-04_scheduler_... (3 runs: seed 0, 1, 2)
# # ... and so on for all combinations
COMMENT


echo "Starting hyperparameter sweep with Cosine Annealing Scheduler..."
echo "Model: ${MODEL}"
echo "Learning rate ranges:"
echo "  1. 0.01 -> 0.001"
echo "  2. 0.001 -> 0.0001"
echo "  3. 0.0001 -> 0.00001"
echo "Lambda regs: ${LAMBDA_REGS}"
echo "Seeds: ${SEEDS}"
echo ""


# Run three separate training configurations with paired LR start/end values
# Each configuration will run all combinations of lambda_reg and seeds

# Configuration 1: LR 0.01 -> 0.001
echo "Running Configuration 1: LR 0.01 -> 0.001"
python train.py -m \
    model=${MODEL} \
    training.use_scheduler=true \
    training.learning_rate=0.01 \
    scheduler.eta_min=0.001 \
    training.lambda_reg=${LAMBDA_REGS} \
    training.n_epochs=${EPOCHS} \
    wandb.project=${PROJECT_NAME} \
    seed=${SEEDS}

# Configuration 2: LR 0.001 -> 0.0001
echo "Running Configuration 2: LR 0.001 -> 0.0001"
python train.py -m \
    model=${MODEL} \
    training.use_scheduler=true \
    training.learning_rate=0.001 \
    scheduler.eta_min=0.0001 \
    training.lambda_reg=${LAMBDA_REGS} \
    training.n_epochs=${EPOCHS} \
    wandb.project=${PROJECT_NAME} \
    seed=${SEEDS}

# Configuration 3: LR 0.001 -> 0.0001
echo "Running Configuration 3: LR 0.01 -> 0.0001"
python train.py -m \
    model=${MODEL} \
    training.use_scheduler=true \
    training.learning_rate=0.01 \
    scheduler.eta_min=0.0001 \
    training.lambda_reg=${LAMBDA_REGS} \
    training.n_epochs=${EPOCHS} \
    wandb.project=${PROJECT_NAME} \
    seed=${SEEDS}

