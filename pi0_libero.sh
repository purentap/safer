MODEL="lstm_img"
LAMBDA_REGS=""1e-3,1e-2,1e-1,1"" #
SEEDS="0,1,2"
EPOCHS="200"
BATCH_SIZE="64"
LEARNING_RATES="1e-5,3e-5,1e-4,3e-4,1e-3" #1e-5, 1e-4,3e-4,1e-3,1e-5
DATASET="pi0-libero"

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