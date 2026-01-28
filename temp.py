import pickle

path = "/mnt/mahzen/puren/research_data/pi0fast_droid/rollouts_with_img_embeddings/pi0fast_droid_0510_all/rollouts_all/20250510_rollouts/20250510_task12_place_the_pink_cup_to_the_right_of_the_blue_cup/img_embed_extracted/task_10--ep_0.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print(data)
