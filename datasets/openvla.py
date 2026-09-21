from __future__ import annotations

from .base import BaseDatasetHandler
import os
import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm
import natsort
from data import RolloutData
    
class OpenVLADatasetHandler(BaseDatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
    def load_rollouts(self, path=None):
        if path is None:
            path = self.cfg.data_root
        all_rollouts = []
        #pkl_files = glob.glob(os.path.join(path, "*.pkl"))
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV rollout files found directly in {path}")
        cntr = 0 
        for csv_file in tqdm(csv_files, desc="Loading data"):
            episode_embeddings = []
            cntr+=1
            pkl_file = csv_file.replace(".csv", ".pkl")
            
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            hidden_states = data[self.cfg.dataset.hidden_feat_name]
            token_idx = round((hidden_states.shape[-2] - 1) * 1)

            # Convert torch tensor to numpy array
            if isinstance(hidden_states, torch.Tensor):
                action_embeddings = hidden_states[..., token_idx, :].detach().cpu().float().numpy()
            else:
                action_embeddings = hidden_states[..., token_idx, :]
            action_embeddings = torch.tensor(action_embeddings, dtype=torch.float32)
            
            episode_embeddings = data[self.cfg.dataset.img_embedding_name]
            episode_embeddings = torch.stack(episode_embeddings)
            episode_embeddings = episode_embeddings.squeeze(1)
            episode_embeddings = episode_embeddings.to(torch.float32)

            rollout_data = RolloutData(episode_embeddings, action_embeddings, data["episode_success"], data["task_id"], data["eposide_idx"])

            all_rollouts.append(rollout_data)
            f.close()


        
            
        #all_rollouts = np.array(all_rollouts)
        #sort rollouts by task_id and episode_idx
        #all_rollouts= sorted(all_rollouts, key=lambda x: (x.get_task_id(), x.get_episode_idx()))
        all_rollouts = self.set_task_min_step(all_rollouts)

        return all_rollouts


