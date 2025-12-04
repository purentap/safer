from utils import load_data
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

data_path = "/mnt/mahzen/puren/open_vla_data_all/rollouts/single-foward/libero_10"
save_folder = "./visualizations"
rollouts = load_data(data_path)
rollouts= sorted(rollouts, key=lambda x: (x.get_task_id(), x.get_episode_idx()))
print("Data loaded successfully")

feats, labels, rollout_indices = [], [], []
feat_skip = 1
for i,r in enumerate(rollouts):
    feat = r.get_img_embeddings()[::feat_skip]
    print(feat.shape)
    feats.append(feat)
    

    labels.append(np.ones(feat.shape[0]) * (1 - r.get_episode_success()))
    rollout_indices.append(np.ones(feat.shape[0]) * i)


    
feats = np.concatenate(feats, axis=0)
labels = np.concatenate(labels, axis=0)
rollout_indices = np.concatenate(rollout_indices, axis=0)
print(f"feats: {feats.shape} {feats.dtype}")
print(f"labels: {labels.shape} {labels.dtype}")
print(f"rollout_indices: {rollout_indices.shape} {rollout_indices.dtype}")

# Visualize the embeddings
projector = TSNE(n_components=2)

feats_projected = projector.fit_transform(feats)
print(f"feats_projected: {feats_projected.shape} {feats_projected.dtype}")

'''Visualize projected features and save as images'''
# Compute the color vector
task_ids = []
colors_by_success = []
for i, r in enumerate(rollouts):
    feat_proj = feats_projected[rollout_indices == i]
    task_ids.append(np.ones(feat_proj.shape[0]) * r.get_task_id())
    if r.get_episode_success() == 0:
        colors_by_success.append(np.linspace(0.6, 1, feat_proj.shape[0]))
    else:
        colors_by_success.append(np.zeros(feat_proj.shape[0]))
colors_by_success = np.concatenate(colors_by_success, axis=0) 
task_ids = np.concatenate(task_ids, axis=0)

# Plot the features and save as images
# Colored by task success



plt.figure(dpi=200)
plt.scatter(
    feats_projected[:, 0], feats_projected[:, 1], 
    c=colors_by_success, cmap='coolwarm', s=0.5, alpha=0.5,
)
plt.axis("off"); plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{save_folder}/feats_vis_skip{feat_skip}-succ.png", bbox_inches='tight')
plt.close()



custom_colors = [
    '#98df8a',  # light green
    '#c5b0d5',  # lavender
    '#8c564b',  # brown
    '#ff7f0e',  # orange
    '#9467bd',  # purple
    '#bcbd22',  # olive
    '#7f7f7f',  # gray
    '#e377c2',  # pink
    '#2ca02c',  # green
    '#c49c94',  # tan
]
cmap_task_ids = mpl.colors.ListedColormap(custom_colors)

plt.figure(dpi=200)
plt.scatter(
    feats_projected[:, 0], feats_projected[:, 1], 
    c=task_ids, cmap=cmap_task_ids, s=0.5, alpha=0.7,
)
plt.axis("off"); plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{save_folder}/feats_vis_skip{feat_skip}-taskid.png", bbox_inches='tight')
plt.close()