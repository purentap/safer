from torch.utils.data import Dataset
import torch
import numpy as np
class RolloutData:
    def __init__(self, img_embeddings, action_embeddings, episode_success, task_id, episode_idx, task_description=None, mp4_path=None):
        self.img_embeddings = img_embeddings
        self.action_embeddings = action_embeddings
        self.episode_success = episode_success
        self.task_id = task_id
        self.episode_idx = episode_idx
        self.task_description = task_description
        self.mp4_path = mp4_path
    def get_img_embeddings(self):
        return self.img_embeddings
    def get_action_embeddings(self):
        return self.action_embeddings
    def get_episode_success(self):
        return self.episode_success
    
    def get_task_id(self):
        return self.task_id
    
    def get_episode_idx(self):
        return self.episode_idx

class RolloutDataset(Dataset):

    def __init__(self, rollouts):
        self.rollouts = rollouts
        self.padded_features, self.valid_masks, labels = pad_rollouts(rollouts, embedding_type="img")
        self.padded_action_embeddings, valid_masks_action, _ = pad_rollouts(rollouts, embedding_type="action")
        self.success_labels = torch.from_numpy(labels)

        # Weigh the loss by the frequency of success/failure
        freq_0 = (sum([r.get_episode_success() == 0 for r in rollouts]) + 1) / len(rollouts) #fail
        freq_1 = (sum([r.get_episode_success() == 1 for r in rollouts]) + 1) / len(rollouts) #success
        self.weights = [1./(freq_0), 1./(freq_1)]
    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        data = {"img_embeddings": self.padded_features[idx],
            "action_embeddings": self.padded_action_embeddings[idx],
            "valid_masks": self.valid_masks[idx],
            "success_labels": self.success_labels[idx]}
        return data
    
    def to(self, device):
        """Move dataset tensors to the specified device."""
        self.padded_features = self.padded_features.to(device)
        self.valid_masks = self.valid_masks.to(device)
        self.success_labels = self.success_labels.to(device)
        return self
    
def pad_rollouts(rollouts, embedding_type="img"):
    if embedding_type == "img":
        all_embeddings = [r.get_img_embeddings() for r in rollouts]
    elif embedding_type == "action":
        all_embeddings = [r.get_action_embeddings() for r in rollouts]
    else:
        raise ValueError(f"Invalid embedding type: {embedding_type}")

    labels = np.array([r.get_episode_success() for r in rollouts])
    
    #determine padding dimensions 
    max_length = max(seq.shape[0] for seq in all_embeddings)
    hidden_dim = all_embeddings[0].shape[-1]
    batch_size = len(all_embeddings)

    # Infer dtype and device from the first sequence
    dtype = all_embeddings[0].dtype

    # Pre-allocate output tensors
    padded_features = torch.zeros(
        (batch_size, max_length, hidden_dim),
        dtype=dtype
    )
    
    padding_masks = torch.ones(
        (batch_size, max_length),
        dtype=torch.float32
    )
    
    for i, seq in enumerate(all_embeddings):
        padded_features[i, :seq.shape[0]] = seq
        padding_masks[i, :seq.shape[0]] = 0
    
    valid_masks = (1 - padding_masks)

    return padded_features, valid_masks, labels
