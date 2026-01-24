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

class Pi0FastLiberoDatasetHandler(BaseDatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def load_rollouts(self):
        all_rollouts = []
        print("***************** Loading rollouts *****************")
        dataset_path = self.cfg.dataset.path
        
        env_records_folder = os.path.join(dataset_path, "env_records")
        policy_records_folder = os.path.join(dataset_path, "policy_records")

        env_record_paths = glob.glob(os.path.join(env_records_folder, "*.pkl"))
        policy_record_paths = glob.glob(os.path.join(policy_records_folder, "*meta.pkl"))

        env_record_paths = natsort.natsorted(env_record_paths)
        policy_record_paths = natsort.natsorted(policy_record_paths)

        all_rollouts = []
    
        policy_step = 0
        for env_record_path in tqdm(env_record_paths, desc="Loading rollouts"):
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

            hidden_states = []
            action_vectors = []
            image_embeddings = []
            for policy_record in policy_records:
                hidden_state = policy_record[self.cfg.dataset.hidden_feat_name]
                image_embedding = policy_record[self.cfg.dataset.img_embedding_name]
                # handle the token dimension
                # (n_tokens, dim_feat) -> (dim_feat)
                hidden_state = hidden_state.mean(axis=-2) #taken from best reproduced result (also paper)
                image_embedding = image_embedding.squeeze(0)
                hidden_states.append(hidden_state)
                image_embeddings.append(image_embedding)
            
            hidden_states = np.stack(hidden_states, axis=0).astype(np.float32)
            hidden_states = torch.from_numpy(hidden_states) # (n_steps, hidden_dim)
            image_embeddings = np.stack(image_embeddings, axis=0).astype(np.float32)
            image_embeddings = torch.from_numpy(image_embeddings) # (n_steps, hidden_dim)
            rollout_data = RolloutData(image_embeddings, hidden_states, env_record["episode_success"], env_record["task_id"], env_record["episode_idx"])
            all_rollouts.append(rollout_data)

        all_rollouts = self.set_task_min_step(all_rollouts)

        return all_rollouts