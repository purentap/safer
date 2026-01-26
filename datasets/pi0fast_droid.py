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

TASK_DESC_TO_STANDARD_LENGTH = {
    "close the drawer": 25,
}

class Pi0FastDroidDatasetHandler(BaseDatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def ensure_task_sr_within(self, all_rollouts, sr_lower_bound, sr_upper_bound, max_rollouts_per_task):
        '''
        Ensure that the success rate (sr) of each task is within the given bounds.
        If not, remove rollouts of that task to achieve the desired success rate.

        if number of rollouts for a task is more than max_rollouts_per_task,
        randomly remove rollouts to keep the number of rollouts within the limit.
        During this removing process, remove the balanced number of successes and failures, 
        to keep the success rate to be the same. 

        Parameters:
        - all_rollouts: List of Rollout objects, each with task_id and episode_success attributes.
        - sr_lower_bound: Optional[float], minimum desired success rate (inclusive).
        - sr_upper_bound: Optional[float], maximum desired success rate (inclusive).
        - max_rollouts_per_task: Optional[int], maximum number of rollouts to keep per task.

        Returns:
        - A filtered list of Rollout. 
        '''
        if sr_lower_bound is not None and sr_upper_bound is not None and sr_lower_bound > sr_upper_bound:
            raise ValueError("sr_lower_bound must be less than or equal to sr_upper_bound")
    
        # Group rollouts by task_id
        task_to_indices = {}
        for idx, rollout in enumerate(all_rollouts):
            task_to_indices.setdefault(rollout.task_id, []).append(idx)

        keep_indices = set(range(len(all_rollouts)))

        for task_id, indices in task_to_indices.items():
            # Separate successes and failures
            successes = [i for i in indices if all_rollouts[i].episode_success == 1]
            failures = [i for i in indices if all_rollouts[i].episode_success == 0]
            total = len(successes) + len(failures)
            if total == 0:
                continue
            current_sr = len(successes) / total

            # Enforce lower bound by dropping failures
            if sr_lower_bound is not None and current_sr < sr_lower_bound and failures:
                # x failures to remove: ceil(total - successes / sr_lower_bound)
                x = int(np.ceil(total - len(successes) / sr_lower_bound))
                x = min(x, len(failures))
                to_remove = list(np.random.choice(failures, size=x, replace=False))
                keep_indices.difference_update(to_remove)
                failures = [i for i in failures if i not in to_remove]
                total -= x
                current_sr = len(successes) / total if total > 0 else 0

            # Enforce upper bound by dropping successes
            if sr_upper_bound is not None and current_sr > sr_upper_bound and successes:
                # y successes to remove: ceil((successes - sr_upper_bound * total) / (1 - sr_upper_bound))
                y = int(np.ceil((len(successes) - sr_upper_bound * total) / (1 - sr_upper_bound)))
                y = min(y, len(successes))
                to_remove = list(np.random.choice(successes, size=y, replace=False))
                keep_indices.difference_update(to_remove)
                successes = [i for i in successes if i not in to_remove]
                total -= y
                current_sr = len(successes) / total if total > 0 else 0

            # Enforce max_rollouts_per_task by balanced removal
            if max_rollouts_per_task is not None and total > max_rollouts_per_task:
                extra = total - max_rollouts_per_task
                s_count = len(successes)
                f_count = len(failures)
                # Compute how many successes/failures to drop (proportional to counts)
                if total > 0:
                    s_remove = int(np.round(extra * s_count / total))
                else:
                    s_remove = 0
                s_remove = min(s_remove, s_count)
                f_remove = extra - s_remove
                f_remove = min(f_remove, f_count)
                # Adjust if rounding short
                removed = s_remove + f_remove
                if removed < extra:
                    remaining = extra - removed
                    # Try removing from failures first
                    add_f = min(remaining, f_count - f_remove)
                    f_remove += add_f
                    remaining -= add_f
                    if remaining > 0:
                        add_s = min(remaining, s_count - s_remove)
                        s_remove += add_s
                        remaining -= add_s
                 # Sample indices to remove
                to_remove_s = list(np.random.choice(successes, size=s_remove, replace=False)) if s_remove > 0 else []
                to_remove_f = list(np.random.choice(failures, size=f_remove, replace=False)) if f_remove > 0 else []
                to_remove = to_remove_s + to_remove_f
                keep_indices.difference_update(to_remove)
                successes = [i for i in successes if i not in to_remove_s]
                failures = [i for i in failures if i not in to_remove_f]
                total -= (s_remove + f_remove)
                current_sr = len(successes) / total if total > 0 else 0
        # Return filtered rollouts in original order
        return [all_rollouts[i] for i in sorted(keep_indices)]


    def compute_task_meta(self, all_rollouts):
        task_ids = set([r.task_id for r in all_rollouts])
        task_ids = sorted(list(task_ids))
        task_metas = {}

        for task_id in task_ids:
            task_rollouts = [r for r in all_rollouts if r.task_id == task_id]
            task_desc = task_rollouts[0].task_description
            n_rollouts = len(task_rollouts)
            n_success = sum([r.episode_success for r in task_rollouts])
            success_rate = n_success / n_rollouts if n_rollouts > 0 else 0.0
            
            task_metas[task_id] = {
                "n_rollouts": n_rollouts,
                "success_rate": success_rate,
                "task_desc": task_desc,
                "task_id": task_id,
            }

        return task_metas
    def print_task_rollout_stats(self, all_rollouts, task_metas = None, print_rollout_lengths = False):
        if task_metas is None:
            task_metas = self.compute_task_meta(all_rollouts)

        for task_id, task_meta in task_metas.items():
            task_desc = task_meta["task_desc"]
            success_rate = task_meta["success_rate"]
            n_rollouts = task_meta["n_rollouts"]
            print(f"Task ID {task_id}: {n_rollouts} rollouts, SR: {success_rate:.2f}, task_desc: {task_desc}")

            if print_rollout_lengths:
                # Plot the distribution of the lengths of the rollouts
                task_rollouts = [r for r in all_rollouts if r.task_id == task_id]
                rollout_lengths = [len(r.action_embeddings) for r in task_rollouts]
                unique, counts = np.unique(rollout_lengths, return_counts=True)
                sort_index = np.argsort(unique)[::-1]
                unique = unique[sort_index]
                counts = counts[sort_index]
                print("Top 5 rollout lengths:", end=" ")
                for length, count in zip(unique[:5], counts[:5]):
                    print(f"Len {length} count {count}; ", end="")
                print("\n")
        


    def load_rollouts_from_root(self, data_folder):
        env_record_paths = glob.glob(os.path.join(self.cfg.dataset.path,data_folder, "env_records", "*.pkl"))
        ep_record_paths = glob.glob(os.path.join(self.cfg.dataset.path,data_folder, "img_embed_extracted", "*.pkl"))
        env_record_paths = natsort.natsorted(env_record_paths)
        ep_record_paths = natsort.natsorted(ep_record_paths)

        all_rollouts = []
        for env_path, ep_path in zip(env_record_paths, ep_record_paths):
            env_record = pickle.load(open(env_path, "rb"))
            ep_record = pickle.load(open(ep_path, "rb"))
            mp4_path = env_path.replace("meta.pkl", "external_left.mp4")
            
            hidden_states = []
            image_embeddings = []

            for ts in ep_record:
                hidden_state = ts[self.cfg.dataset.hidden_feat_name]
                img_embedding = ts[self.cfg.dataset.img_embedding_name]
                hidden_state = hidden_state.mean(axis=-2) #taken from paper, get mean of the hidden states
                img_embedding = img_embedding.squeeze(0)

                hidden_states.append(hidden_state)
                image_embeddings.append(img_embedding)

            
            hidden_states = np.stack(hidden_states, axis=0).astype(np.float32)
            hidden_states = torch.from_numpy(hidden_states) # (n_steps, hidden_dim)
            image_embeddings = np.stack(image_embeddings, axis=0).astype(np.float32)
            image_embeddings = torch.from_numpy(image_embeddings) # (n_steps, hidden_dim)
 
            
            rollout_data = RolloutData(image_embeddings, 
                                        hidden_states, 
                                        int(env_record["episode_success"]), 
                                        env_record["task_id"], 
                                        env_record["episode_idx"], 
                                        task_description=env_record["task_description"],
                                        mp4_path=mp4_path)
            all_rollouts.append(rollout_data)

            #print(ep_record.keys())
       
        return all_rollouts

    def load_rollouts(self):
        
        all_data_folders= self.cfg.dataset.data_path
        #print(all_data_folders)
        all_rollouts = []
        for path in tqdm(all_data_folders, desc="Loading rollouts"):
            rollouts = self.load_rollouts_from_root(path)
            all_rollouts.extend(rollouts)
        print(len(all_rollouts))

         # Redo the task_id based on task_description for all rollouts
        task_descs = list(set([r.task_description for r in all_rollouts]))
        task_descs = sorted(task_descs)
        task_desc_to_id = {task_desc: i for i, task_desc in enumerate(task_descs)}
        for r in all_rollouts:
            r.task_id = task_desc_to_id[r.task_description]
        print(f"Found {len(task_descs)} unique tasks with total {len(all_rollouts)} rollouts")
        self.print_task_rollout_stats(all_rollouts, print_rollout_lengths=True)

        #They only keep the rollouts with max length in each task
        indices_to_keep = []
        task_ids = sorted(list[int](task_desc_to_id.values()))
        for task_id in task_ids:
            task_rollouts = [r for r in all_rollouts if r.task_id == task_id]
            task_desc = task_rollouts[0].task_description

            if task_desc in TASK_DESC_TO_STANDARD_LENGTH:
                desired_length = TASK_DESC_TO_STANDARD_LENGTH[task_desc]
            else:
                desired_length = max([len(r.action_embeddings) for r in task_rollouts])
            indices_to_keep.extend(
                    [i for i, r in enumerate(all_rollouts) 
                    if r.task_id == task_id and len(r.action_embeddings) == desired_length]
                )
        indices_to_keep = sorted(indices_to_keep)
        all_rollouts = [all_rollouts[i] for i in indices_to_keep]
        print(f"Only keeping the rollouts of the max length in each task, {len(all_rollouts)} rollouts remain")

        all_rollouts = self.set_task_min_step(all_rollouts)

        # Compute the meta information for each task
        task_metas = self.compute_task_meta(all_rollouts)
        self.print_task_rollout_stats(all_rollouts, task_metas)

        # Remove some rollouts to ensure the success rate of each task is within the given bounds
        all_rollouts = self.ensure_task_sr_within(all_rollouts, self.cfg.dataset.adjust_sr_min, self.cfg.dataset.adjust_sr_max, self.cfg.dataset.max_rollouts_per_task)

        task_metas = self.compute_task_meta(all_rollouts)
    
        # Print the final statistics of the task descriptions and success rates
        print("="*20)
        print("Final statistics of the task descriptions and success rates:")
        self.print_task_rollout_stats(all_rollouts, task_metas)

        print(len(all_rollouts))
        return all_rollouts