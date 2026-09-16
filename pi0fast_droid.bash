#!/bin/bash

MODEL="fusion_lstmv2"
LAMBDA_REGS="1e-3,1e-2,1e-1" #
SEEDS="0,1,2,3,4"
EPOCHS="500"
BATCH_SIZE="64"
LEARNING_RATES="1e-4,3e-4,1e-3,3e-3" 
DATASET="pi0fast_droid"


python train.py -m \
    model="${MODEL}" \
    dataset="${DATASET}" \
    training.use_scheduler=false \
    training.learning_rate=${LEARNING_RATES} \
    training.batch_size="${BATCH_SIZE}" \
    training.lambda_reg="${LAMBDA_REGS}" \
    wandb.project="${MODEL}_${DATASET}_batch_${BATCH_SIZE}_epochs_${EPOCHS}" \
    wandb.enabled=true \
    seed=${SEEDS} \
    training.n_epochs=${EPOCHS}

# MODEL="fusion_lstmv2"
# LAMBDA_REGS=(1e-3 1e-2 1e-1)
# SEEDS=(0 1 2 3 4)
# EPOCHS="500"
# BATCH_SIZE="64"
# LEARNING_RATES=(1e-4 3e-4 1e-3 3e-3)
# DATASET="pi0fast_droid"

# for lr in "${LEARNING_RATES[@]}"; do
#   for lam in "${LAMBDA_REGS[@]}"; do
#     for seed in "${SEEDS[@]}"; do
#       python train.py \
#         model="${MODEL}" \
#         dataset="${DATASET}" \
#         training.use_scheduler=false \
#         training.learning_rate="${lr}" \
#         training.batch_size="${BATCH_SIZE}" \
#         training.lambda_reg="${lam}" \
#         wandb.project="LAYERNORM_new_script_${MODEL}_${DATASET}_batch_${BATCH_SIZE}_epochs_${EPOCHS}" \
#         wandb.enabled=true \
#         seed="${seed}" \
#         training.n_epochs="${EPOCHS}"
#     done
#   done
# done