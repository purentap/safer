from __future__ import annotations

from typing import Any, Dict

from data import RolloutDataset
import numpy as np
import torch


class BaseDatasetHandler:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def load_rollouts(self) -> Any:
        # override in subclasses
        raise NotImplementedError("Subclasses must implement load_rollouts()")
    def split_rollouts(self, rollouts):
        #taken from safe-vla codebase
        # Split rollouts into seen and unseen tasks
        task_ids = list(set([r.get_task_id() for r in rollouts]))
        n_unseen = round(self.cfg.dataset.unseen_task_ratio * len(task_ids))
        n_seen = len(task_ids) - n_unseen

        np.random.shuffle(task_ids)
        seen_task_ids = task_ids[:n_seen]
        unseen_task_ids = task_ids[n_seen:]
        rollouts_by_split_name = self.split_rollouts_by_seen_unseen(rollouts, seen_task_ids, unseen_task_ids)
        return rollouts_by_split_name
        #return train_rollouts, val_rollouts

    def split_rollouts_by_seen_unseen(self, rollouts, seen_task_ids, unseen_task_ids):
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
            n_train_rollouts = int(self.cfg.dataset.seen_train_ratio * len(task_rollouts))
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

    def set_task_min_step(self,rollouts):
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