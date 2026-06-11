import torch
from model import FusionLSTMModel_v2
from datasets import get_dataset_handler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data import RolloutDataset
import time 
import numpy as np
model_path = "/home/enes/puren/research/fail_detection_embeddings/best_model.pth"
cfg_path = "/home/enes/puren/research/fail_detection_embeddings/best_model_cfg.yaml"

cfg = OmegaConf.load(cfg_path)
model = FusionLSTMModel_v2(cfg.model.params)
model.load_state_dict(torch.load(model_path))


model.to("cuda")
model.eval()

dataset_name = cfg.dataset.get("name", None) if hasattr(cfg, "dataset") else None

DatasetHandler = get_dataset_handler(dataset_name)
dataset_handler = DatasetHandler(cfg)

rollouts = dataset_handler.load_rollouts()


print(rollouts[0].get_img_embeddings().dtype)
print(rollouts[0].get_action_embeddings().dtype)