#!/bin/bash
# Using Hydra multi-run for cosine annealing scheduler with different LR ranges

# This script runs three training configurations with cosine annealing:
# 1. LR: 0.01 -> 0.001
# 2. LR: 0.001 -> 0.0001
# 3. LR: 0.0001 -> 0.00001

# Customize these values as needed:
MODEL="lstm_img_action"
LAMBDA_REGS=""1e-3,1e-2,1e-1,1"" #
SEEDS="0,1,2"
EPOCHS="200"
BATCH_SIZE="64"
LEARNING_RATES="1e-5,1e-4,3e-4,1e-3" #1e-5, 1e-4,3e-4,1e-3,1e-5

#for MODEL in "lstm_img" "lstm_img_action" "fusion_lstm"; do
python train.py -m \
    model="${MODEL}" \
    training.use_scheduler=false \
    training.learning_rate=${LEARNING_RATES} \
    training.batch_size="${BATCH_SIZE}" \
    training.lambda_reg="${LAMBDA_REGS}" \
    wandb.project="${MODEL}wDropout_noscheduler_sweep_batch_${BATCH_SIZE}_epochs_${EPOCHS}" \
    wandb.enabled=true \
    seed=${SEEDS} \
    training.n_epochs=${EPOCHS}
#done
# echo "Starting hyperparameter sweep with Cosine Annealing Scheduler..."
# echo "Model: ${MODEL}"
# echo "Batch size: ${BATCH_SIZE}"
# echo "Epochs: ${EPOCHS}"
# echo "Learning rate ranges:"
# echo "  1. 0.01 -> 0.001"
# echo "  2. 0.001 -> 0.0001"
# echo "  3. 0.01 -> 0.0001"
# echo "Lambda regs: ${LAMBDA_REGS}"
# echo "Seeds: ${SEEDS}"
# echo "Project name: ${PROJECT_NAME}"
# echo ""

# # Run three separate training configurations with paired LR start/end values
# # Each configuration will run all combinations of lambda_reg and seeds

# # Configuration 1: LR 0.01 -> 0.001
# echo "Running Configuration 1: LR 0.01 -> 0.001"
# python train.py -m \
#     model="${MODEL}" \
#     training.use_scheduler=true \
#     training.learning_rate=0.01 \
#     scheduler.eta_min=0.001 \
#     training.lambda_reg="${LAMBDA_REGS}" \
#     wandb.project=${PROJECT_NAME} \
#     seed="${SEEDS}"\
#     training.n_epochs="${EPOCHS}" \
#     training.batch_size="${BATCH_SIZE}"

# # Configuration 2: LR 0.001 -> 0.0001
# echo "Running Configuration 2: LR 0.001 -> 0.0001"
# python train.py -m \
#     model="${MODEL}" \
#     training.use_scheduler=true \
#     training.learning_rate=0.001 \
#     scheduler.eta_min=0.0001 \
#     training.lambda_reg="${LAMBDA_REGS}" \
#     wandb.project=${PROJECT_NAME} \
#     seed="${SEEDS}" \
#     training.n_epochs="${EPOCHS}" \
#     training.batch_size="${BATCH_SIZE}"

# # Configuration 3: LR 0.01 -> 0.0001
# echo "Running Configuration 3: LR 0.01 -> 0.0001"
# python train.py -m \
#     model="${MODEL}" \
#     training.use_scheduler=true \
#     training.learning_rate=0.01 \
#     scheduler.eta_min=0.0001 \
#     training.lambda_reg="${LAMBDA_REGS}" \
#     wandb.project=${PROJECT_NAME} \
#     seed="${SEEDS}" \
#     training.n_epochs="${EPOCHS}" \
#     training.batch_size="${BATCH_SIZE}"

# echo "All training configurations completed!"
