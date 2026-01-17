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
data_path = "/mnt/mahzen/puren/open_vla_data_all/rollouts/single-foward/libero_10"

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

def normalize_rollouts_hidden_states(
    rollouts,
):
    """
    Normalize the hidden states of the rollouts to have zero mean and unit variance.
    This is done in place, modifying the original rollouts.
    """
    train_rollouts = rollouts["train"]
    # Stack all hidden states into a single tensor
    train_img_embeddings = torch.cat([r.img_embeddings for r in train_rollouts], dim=0)
    train_action_embeddings = torch.cat([r.action_embeddings for r in train_rollouts], dim=0)
    # Compute mean and std
    train_mean_img = train_img_embeddings.mean(dim=0)
    train_std_img = train_img_embeddings.std(dim=0)
    train_mean_action = train_action_embeddings.mean(dim=0)
    train_std_action = train_action_embeddings.std(dim=0)
    # Normalize each rollout's hidden states
    for k,v in rollouts.items():
        for r in v:
            r.img_embeddings = (r.img_embeddings - train_mean_img) / train_std_img
            r.action_embeddings = (r.action_embeddings - train_mean_action) / train_std_action
        rollouts[k] = v
    return rollouts

def load_data_pi0(path):
    env_records_folder = os.path.join(path, "env_records")
    policy_records_folder = os.path.join(path, "policy_records")
    
    env_record_paths = glob.glob(os.path.join(env_records_folder, "*.pkl"))
    policy_record_paths = glob.glob(os.path.join(policy_records_folder, "*meta.pkl"))
    

    env_record_paths = natsort.natsorted(env_record_paths)
    policy_record_paths = natsort.natsorted(policy_record_paths)
    all_rollouts = []
    
    policy_step = 0

    for env_record_path in tqdm(env_record_paths):
        # Load the meta data from the env record
        env_record = pickle.load(open(env_record_path, "rb"))
        mp4_path = env_record_path.replace(".pkl", ".mp4")

        # Load hidden features from corresponding policy records
        model_infer_times = env_record["model_infer_times"]
        policy_records = []

        for i in range(model_infer_times):
            policy_record_path = policy_record_paths[policy_step]
            policy_records.append(pickle.load(open(policy_record_path, "rb")))
            policy_step += 1
            
        
        # Extract hidden states and actions from policy records
        hidden_states = []
        action_vectors = []
        image_embeddings = []

        for policy_record in policy_records:
            
            # hidden_state shape: (n_diff_steps, n_pred_horizon, dim_feats)
            hidden_state = policy_record["pre_velocity"]

            # handle the pred_horizon dimension
            # we use the first dimension since its reported it works best.
            horizon_idx = 0
            token_idx = round((hidden_state.shape[-2] - 1) * horizon_idx)
            hidden_state = hidden_state[:, token_idx, :]
            # handle the diff_steps dimension
            # we use the last dimension since its reported it works best. 
            diff_idx = 1
            diff_step_idx = round((hidden_state.shape[-2] - 1) * diff_idx)

            hidden_state = hidden_state[..., diff_step_idx, : ]

            hidden_states.append(hidden_state)

            image_embedding = policy_record["prefix_tokens"].squeeze(0)
            action = policy_record["actions"].reshape(-1)

            image_embeddings.append(image_embedding)
            action_vectors.append(action)
            
        hidden_states = np.stack(hidden_states, axis=0).astype(np.float32)
        hidden_states = torch.from_numpy(hidden_states) # (n_steps, hidden_dim)
        action_vectors = np.stack(action_vectors, axis=0).astype(np.float32)
        action_vectors = torch.from_numpy(action_vectors) # (n_steps, pred_horizon*action_dim)
        image_embeddings = np.stack(image_embeddings, axis=0).astype(np.float32)
        image_embeddings = torch.from_numpy(image_embeddings) # (n_steps, hidden_dim)


        rollout_data = RolloutData(image_embeddings, hidden_states, env_record["episode_success"], env_record["task_id"], env_record["episode_idx"])
        all_rollouts.append(rollout_data)
    all_rollouts = set_task_min_step(all_rollouts)
    return all_rollouts