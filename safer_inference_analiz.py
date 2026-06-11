import torch
from model import FusionLSTMModel_v2, LSTMModel
from datasets import get_dataset_handler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data import RolloutDataset
import time 
import numpy as np
model_path = "/home/enes/puren/research/fail_detection_embeddings/openvla_seed0.pth"
cfg_path = "/home/enes/puren/research/fail_detection_embeddings/openvla_seed0.yaml"
cfg = OmegaConf.load(cfg_path)
#model = FusionLSTMModel_v2(cfg.model.params)
model = LSTMModel(cfg.model.params)
model.load_state_dict(torch.load(model_path))


model.to("cuda")
model.eval()

print(model)

dataset_name = cfg.dataset.get("name", None) if hasattr(cfg, "dataset") else None

DatasetHandler = get_dataset_handler(dataset_name)
dataset_handler = DatasetHandler(cfg)

rollouts = dataset_handler.load_rollouts()

print(rollouts[0].get_img_embeddings().dtype)
print(rollouts[0].get_action_embeddings().dtype)

splitted_rollouts = dataset_handler.split_rollouts(rollouts)
# Construct datasets and dataloaders from the rollouts
dataset_by_split_name = {
    k: RolloutDataset(v) 
    for k, v in splitted_rollouts.items()
}
#if cfg.training.normalize_hidden_states:
#    dataset_by_split_name = utils.normalize_rollouts_hidden_states(dataset_by_split_name)

dataloader_by_split_name = {
    k: DataLoader(
        v, 
        batch_size=cfg.training.batch_size, 
        shuffle="train" in k, 
        num_workers=0)
    for k, v in dataset_by_split_name.items()
}


dataloader = dataloader_by_split_name["val_seen"]
batch = next(iter(dataloader))
B = batch["action_embeddings"].shape[0]
print("label: " , batch["success_labels"][0])
#single_rollout = {k: v[0] for k, v in batch.items()}
actions= batch["action_embeddings"].to("cuda")
imgs = batch["img_embeddings"].to("cuda")
print(actions.shape)
print(imgs.shape)



# -------------------
# Warm-up
# -------------------
warmup_iters = 100
measure_iters = 1000
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    action = actions[0].unsqueeze(0)
    img = imgs[0].unsqueeze(0)

    for _ in range(warmup_iters):
        _ = model(img, action)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    
    # -------------------
    # Measurement
    # -------------------
    timings = []

    for i in range(measure_iters):
        action = actions[i % B].unsqueeze(0)
        img = imgs[i % B].unsqueeze(0)

        torch.cuda.synchronize()
        starter.record()
        _ = model(img, action)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings.append(curr_time)  # ms

    timings = np.array(timings)

    print(f"Mean latency:   {timings.mean():.3f} ms")
    print(f"Std latency:    {timings.std():.3f} ms")
    print(f"Median latency: {np.median(timings):.3f} ms")
    print(f"P95 latency:    {np.percentile(timings, 95):.3f} ms")

print(timings)