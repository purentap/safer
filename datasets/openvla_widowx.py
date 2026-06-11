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
import natsort


class OpenVlaWidowxDatasetHandler(BaseDatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def load_rollouts(self):
        all_rollouts = []
        all_csv = glob.glob(f"{self.cfg.dataset.path}*.csv")
        #print(all_csv)
        #pkl_files = glob.glob(os.path.join(self.cfg.dataset.path, "*.pkl"))
        all_csv = natsort.natsorted(all_csv)
        cntr = 0 

        for csv_path in tqdm(all_csv, desc="Loading data"):
            pkl_path = csv_path.replace(".csv", ".pkl")
    
            episode_embeddings = []
            cntr+=1
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            hidden_states = data["hidden_states"]
            img_embeddings=data["img_embeds"]
            #convert list of tensors to numpy array
            #hidden_states = torch.stack(hidden_states).detach().cpu().float().numpy()
            #convert image embeddings list of arrays to torch tensor
            img_embeddings = torch.tensor(np.stack(img_embeddings), dtype=torch.float32)
            hidden_states = torch.stack(hidden_states).to(torch.float32)

            #print(img_embeddings.shape)
            #print(hidden_states.shape)
            token_idx = round((hidden_states.shape[-2] - 1) * 1) #taken from best reproduced result, last dimension is the best
            hidden_states= hidden_states[:, token_idx, :]
            #print(hidden_states.shape)
            rollout_data = RolloutData(img_embeddings, 
                                    hidden_states, 
                                    data["episode_success"], 
                                    data["task_id"], 
                                    data["eposide_idx"],
                                    task_description= data["task_description"],
                                    mp4_path = data["mp4_path"])
            
            all_rollouts.append(rollout_data)
            f.close()
        #all_rollouts= sorted(all_rollouts, key=lambda x: (x.get_task_id(), x.get_episode_idx()))
        all_rollouts = self.set_task_min_step(all_rollouts)
        print(len(all_rollouts))
        return all_rollouts
