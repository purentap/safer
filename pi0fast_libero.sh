MODEL="fusion_lstmv2"
LAMBDA_REGS="1e-3,1e-2,1e-1" #
SEEDS="0,1,2"
EPOCHS="1000"
BATCH_SIZE="64"
LEARNING_RATES="3e-5,1e-5,1e-4,3e-4,1e-3" 
DATASET="pi0fast_libero"


python train.py -m \
    model="${MODEL}" \
    dataset="${DATASET}" \
    training.use_scheduler=false \
    training.learning_rate=${LEARNING_RATES} \
    training.batch_size="${BATCH_SIZE}" \
    training.lambda_reg="${LAMBDA_REGS}" \
    wandb.project="CORRECTEMBEDS_${MODEL}_${DATASET}_batch_${BATCH_SIZE}_epochs_${EPOCHS}" \
    wandb.enabled=true \
    seed=${SEEDS} \
    training.n_epochs=${EPOCHS}