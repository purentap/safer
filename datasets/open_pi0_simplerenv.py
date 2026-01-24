from __future__ import annotations

from .base import BaseDatasetHandler
import os
import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm
import natsort
import json
from data import RolloutData

class OpenPi0SimplerDatasetHandler(BaseDatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def load_rollouts(self):
        all_rollouts = []
        print("***************** Loading rollouts *****************")
        dataset_path = self.cfg.dataset.path
        all_task_names = os.listdir(dataset_path)

        keep_tasks = [
        "widowx_carrot_on_plate",
        "widowx_put_eggplant_in_basket",
        "widowx_spoon_on_towel",
        "widowx_stack_cube",
        "google_robot_move_near_v0",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_place_apple_in_closed_top_drawer",
        ]

        task_names = [t for t in keep_tasks if t in all_task_names]
    
        print(f"Keeping {len(task_names)} tasks: {task_names}")
    
        for task_id, task_name in enumerate(task_names):
            task_path = os.path.join(dataset_path, task_name)
            meta_info_paths = glob.glob(f"{task_path}/*_meta.json")

            for meta_info_path in tqdm(meta_info_paths):

                pkl_path = meta_info_path.replace("_meta.json", ".pkl")
                mp4_path = meta_info_path.replace("_meta.json", ".mp4")

                # Load the raw rollout data
                with open(meta_info_path, "r") as f:
                    meta_info = json.load(f)
                with open(pkl_path, "rb") as f:
                    rollout_raw = pickle.load(f)
                
                # Extract the hidden states and sampled actions
                hidden_states = []
                sampled_actions = []
                image_embeddings = []
                for rollout in rollout_raw:
                    action_embeds = rollout[self.cfg.dataset.hidden_feat_name] # (N, T, H, E)
                    img_embeds = rollout[self.cfg.dataset.image_embed_name] # (1, 1152)

                    # handle the horizon dimension
                    # (N, T, H, E) -> (N, T, E)
                    horizon_idx = 1 #taken from best reproduced result
                    token_idx = round((action_embeds.shape[-2] - 1) * horizon_idx)
                    action_embeds = action_embeds[..., token_idx, :]

                    # handle the diff_steps dimension
                    # (N, T, E) -> (N, E)
                    diff_idx = 1.0 #taken from best reproduced result
                    token_idx = round((action_embeds.shape[-2] - 1) * diff_idx)
                    action_embeds = action_embeds[..., token_idx, :]
                    # k=2
                    # # Determine the number of indices available along the second-to-last dimension.
                    # c = action_embeds.shape[-2]
                    # # Compute k indices uniformly spaced, including the endpoints.
                    # indices = np.linspace(0, c - 1, num=k)
                    # # Convert to integers by rounding.
                    # indices = np.round(indices).astype(int)
                    # # Use these indices to select along the second-to-last dimension.
                    # indexed = action_embeds[..., indices, :]
                    # new_last_dim = indexed.shape[-2] * indexed.shape[-1]
                    # action_embeds = indexed.reshape(*indexed.shape[:-2], new_last_dim)


                    #Only keep embeddings from the last sampled action (used for the actual execution)
                    action_embeds = action_embeds[-1].reshape(-1)
                    img_embeds = img_embeds.squeeze(0)

                    hidden_states.append(action_embeds)
                    image_embeddings.append(img_embeds)
            
                # the loaded action embeddings are already torch tensors (bf16 or float32)
                hidden_states = torch.stack(hidden_states).float() # (T, E)
                image_embeddings = np.stack(image_embeddings, axis=0).astype(np.float32)
                image_embeddings = torch.from_numpy(image_embeddings) # (n_steps, hidden_dim)

  
                rollout_data = RolloutData(image_embeddings, 
                                        hidden_states, 
                                        meta_info["success"],
                                        task_id, 
                                        meta_info["episode_id"])
                
                all_rollouts.append(rollout_data)

        all_rollouts = self.set_task_min_step(all_rollouts)
        return all_rollouts





