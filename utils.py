import glob 
import os
from tqdm import tqdm   
import pickle
import numpy as np
import torch
import random
from data import RolloutData
from model import MLPModel, LSTMModel, FusionLSTMModel
data_path = "/mnt/mahzen/puren/open_vla_data_all/rollouts/single-foward/libero_10"

model_classes = {
    "mlp": MLPModel,
    "LSTMModel": LSTMModel,
    "FusionLSTMModel": FusionLSTMModel
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
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_class}")
    else:
        return None

def set_task_min_step(rollouts):
    '''
    Compute the minimum timestep for each task
    This operation modifies the input rollouts in place.
    '''
    task_ids = list(set([r.get_task_id() for r in rollouts]))
    for task_id in task_ids:
        task_rollouts = [r for r in rollouts if r.get_task_id() == task_id]
        min_timestep = min([r.get_img_embeddings().shape[0] for r in task_rollouts])
        for r in rollouts:
            if r.get_task_id() == task_id:
                r.task_min_step = min_timestep
                
    return rollouts

def split_rollouts(rollouts, split_ratio=0.8):
    #taken from safe-vla codebase
    # Split rollouts into seen and unseen tasks
    task_ids = list(set([r.get_task_id() for r in rollouts]))
    n_unseen = round(0.3 * len(task_ids))
    n_seen = len(task_ids) - n_unseen

    np.random.shuffle(task_ids)
    seen_task_ids = task_ids[:n_seen]
    unseen_task_ids = task_ids[n_seen:]
    rollouts_by_split_name = split_rollouts_by_seen_unseen(rollouts, seen_task_ids, unseen_task_ids)
    return rollouts_by_split_name
    #return train_rollouts, val_rollouts

def split_rollouts_by_seen_unseen(rollouts, seen_task_ids, unseen_task_ids):
    #taken from safe-vla codebase

    print(f"Seen tasks: {seen_task_ids}, Unseen tasks: {unseen_task_ids}")
    seen_rollouts = [r for r in rollouts if r.get_task_id() in seen_task_ids]
    unseen_rollouts = [r for r in rollouts if r.get_task_id() in unseen_task_ids]

    # Split the rollouts for training and evaluation
    train_rollouts = []
    val_seen_rollouts = []

    for task_id in seen_task_ids:
        task_rollouts = [r for r in seen_rollouts if r.get_task_id() == task_id]    
        # Split the seen tasks into training and val_seen sets
        permuted_indices = torch.randperm(len(task_rollouts))
        n_train_rollouts = int(0.6 * len(task_rollouts))
        train_rollouts += [task_rollouts[i] for i in permuted_indices[:n_train_rollouts]]
        val_seen_rollouts += [task_rollouts[i] for i in permuted_indices[n_train_rollouts:]]
    val_unseen_rollouts = unseen_rollouts

    rollouts_by_split_name = {
        "train": train_rollouts,
        "val_seen": val_seen_rollouts,
        "val_unseen": val_unseen_rollouts,
    }
    
    if len(val_unseen_rollouts) == 0:
        del rollouts_by_split_name["val_unseen"]
    if len(val_seen_rollouts) == 0:
        del rollouts_by_split_name["val_seen"]
    
    for split, rollouts in rollouts_by_split_name.items():
        n_success = sum([r.episode_success for r in rollouts])
        n_fail = len(rollouts) - n_success
        print(f"{split}: {len(rollouts)} rollouts, {n_success} success, {n_fail} fail")

    return rollouts_by_split_name
def load_data(path):
    all_rollouts = []
    pkl_files = glob.glob(os.path.join(path, "*.pkl"))
    cntr = 0 
    for pkl_file in tqdm(pkl_files, desc="Loading data"):
        episode_embeddings = []
        cntr+=1
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        #print(data["embeddings_and_attention_masks"][0].keys())

        hidden_states = data["hidden_states"]
        token_idx = round((hidden_states.shape[-2] - 1) * 1)

        # Convert torch tensor to numpy array
        if isinstance(hidden_states, torch.Tensor):
            action_embeddings = hidden_states[:, token_idx, :].detach().cpu().float().numpy()
        else:
            action_embeddings = hidden_states[:, token_idx, :]
        
        '''
        rollout_embeddings = data["embeddings_and_attention_masks"]

        for ts in rollout_embeddings:
            patch_features = ts["patch_features"]
            img_embedding = (patch_features.reshape(-1, patch_features.shape[-1])).mean(axis=0).float().numpy()
            episode_embeddings.append(img_embedding)
        
        episode_embeddings = np.array(episode_embeddings)
        '''
        
        episode_embeddings = data["embeddings"]
        #rollout_data = RolloutData(episode_embeddings, data["episode_success"], data["task_id"], data["eposide_idx"])
        rollout_data = RolloutData(episode_embeddings, action_embeddings, data["success"], data["task_id"], data["episode_idx"])


        #print(episode_embeddings)
        all_rollouts.append(rollout_data)
        f.close()


    
        
    #all_rollouts = np.array(all_rollouts)
    #sort rollouts by task_id and episode_idx
    all_rollouts= sorted(all_rollouts, key=lambda x: (x.get_task_id(), x.get_episode_idx()))
    all_rollouts = set_task_min_step(all_rollouts)

    return all_rollouts

#rollouts = load_data(data_path)
#print(rollouts[0].get_img_embeddings())
#print(labels)
#print(rollouts)

def seed_everything(seed: int) -> None:
    """Set the seed for all random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
