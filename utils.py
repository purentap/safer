import glob 
import os
from tqdm import tqdm   
import pickle
import numpy as np
import torch
import random
import natsort

from data import RolloutData
from model import MLPModel, LSTMModel, FusionLSTMModel, FusionLSTMModel_v2, DoubleLSTMModel

model_classes = {
    "mlp": MLPModel,
    "LSTMModel": LSTMModel,
    "FusionLSTMModel": FusionLSTMModel,
    "FusionLSTMModel_v2": FusionLSTMModel_v2,
    "DoubleLSTMModel": DoubleLSTMModel,
}

optimizer_classes = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD
}
def create_model(cfg):
    model_config = cfg.model
    model_class = model_config.class_name
    model = model_classes[model_class](cfg=model_config.params)
    print(f"Model created successfully: {model}")
    return model

def create_optimizer(model, cfg):
    optimizer_config = cfg.optimizer
    optimizer_class = optimizer_config.type
    if optimizer_class == "Adam":
        optimizer= torch.optim.Adam(model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay)
        print(f"Optimizer created successfully: {optimizer}")
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_class}")
    return optimizer
def create_scheduler(optimizer, cfg):
    if cfg.training.use_scheduler:
        scheduler_config = cfg.scheduler
        scheduler_class = scheduler_config.type
        if scheduler_class == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config.T_max, eta_min=scheduler_config.eta_min)
        elif scheduler_class == "StepLRonPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_config.factor, patience=scheduler_config.patience, min_lr=scheduler_config.min_lr)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_class}")
    else:
        return None

def seed_everything(seed: int) -> None:
    """Set the seed for all random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)