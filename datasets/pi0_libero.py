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


class Pi0LiberoDatasetHandler(BaseDatasetHandler):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.path = cfg.data_root
        self.unseen_task_ratio = cfg.dataset.unseen_task_ratio
        self.seen_train_ratio = cfg.dataset.seen_train_ratio
        self.hidden_feature_dim = cfg.dataset.hidden_feature_dim
        self.image_embedding_dim = cfg.dataset.image_embedding_dim
        self.hidden_feat_name = cfg.dataset.hidden_feat_name
        self.img_embedding_name = cfg.dataset.img_embedding_name
        self.total_input_dim = self.hidden_feature_dim + self.image_embedding_dim
        self.cfg = cfg
    def load_rollouts(self):
        env_records_folder = os.path.join(self.path, "env_records")
        policy_records_folder = os.path.join(self.path, "policy_records")
        
        env_record_paths = glob.glob(os.path.join(env_records_folder, "*.pkl"))
        policy_record_paths = glob.glob(os.path.join(policy_records_folder, "*meta.pkl"))
        if not env_record_paths or not policy_record_paths:
            raise FileNotFoundError(
                f"Expected rollout files in {env_records_folder} and {policy_records_folder}"
            )
        

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
                hidden_state = policy_record[self.cfg.dataset.hidden_feat_name]

                # handle the pred_horizon dimension
                # we use the first dimension since its reported it works best.
                horizon_idx = 0
                token_idx = round((hidden_state.shape[-2] - 1) * horizon_idx)
                hidden_state = hidden_state[:, token_idx, :]
                # handle the diff_steps dimension
                # we use the last dimension since its reported it works best. 
                # diff_idx = 1
                # diff_step_idx = round((hidden_state.shape[-2] - 1) * diff_idx)

                # hidden_state = hidden_state[..., diff_step_idx, : ]
                
                k=2
                # Determine the number of indices available along the second-to-last dimension.
                c = hidden_state.shape[-2]
                # Compute k indices uniformly spaced, including the endpoints.
                indices = np.linspace(0, c - 1, num=k)
                # Convert to integers by rounding.
                indices = np.round(indices).astype(int)
                # Use these indices to select along the second-to-last dimension.
                indexed = hidden_state[..., indices, :]
                new_last_dim = indexed.shape[-2] * indexed.shape[-1]
                hidden_state = indexed.reshape(*indexed.shape[:-2], new_last_dim)
                
                self.cfg.dataset.hidden_feature_dim = hidden_state.shape[0]
                hidden_states.append(hidden_state)

                image_embedding = policy_record[self.cfg.dataset.img_embedding_name].squeeze(0)
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
        all_rollouts = self.set_task_min_step(all_rollouts)
        return all_rollouts
