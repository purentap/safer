MODEL="fusion_lstmv2"
LAMBDA_REGS="1e-3" #
SEEDS="0,1,2,3,4"
EPOCHS="500"
BATCH_SIZE="64"
LEARNING_RATES="3e-4" 
DATASET="pi0fast_droid"


python train.py -m \
    model="${MODEL}" \
    dataset="${DATASET}" \
    training.use_scheduler=false \
    training.learning_rate=${LEARNING_RATES} \
    training.batch_size="${BATCH_SIZE}" \
    training.lambda_reg="${LAMBDA_REGS}" \
    wandb.project="pi0fast_droid_fusion_lstmv2_youdensj" \
    wandb.enabled=true \
    seed=${SEEDS} \
    training.n_epochs=${EPOCHS} \
    training.save_best_model=true \