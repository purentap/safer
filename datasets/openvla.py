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
            path = self.cfg.dataset.path
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
        #all_rollouts= sorted(all_rollouts, key=lambda x: (x.get_task_id(), x.get_episode_idx()))
        all_rollouts = self.set_task_min_step(all_rollouts)

        return all_rollouts


