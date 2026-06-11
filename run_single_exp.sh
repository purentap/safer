MODEL="fusion_lstmv2"
LAMBDA_REGS="1e-3" #
SEEDS="1,4,7"
EPOCHS="500"
BATCH_SIZE="64"
LEARNING_RATES="1e-3" #1e-5, 1e-4,3e-4,1e-3,1e-5
DATASET="openvla_widowx"

python train.py -m \
    model="${MODEL}" \
    dataset="${DATASET}" \
    training.use_scheduler=false \
    training.learning_rate=${LEARNING_RATES} \
    training.batch_size="${BATCH_SIZE}" \
    training.lambda_reg="${LAMBDA_REGS}" \
    wandb.project="openvla_widowx_fusion_lstmv2_youdensj" \
    wandb.enabled=true \
    seed=${SEEDS} \
    training.n_epochs=${EPOCHS} \
