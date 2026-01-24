import pickle
path = "/mnt/mahzen/puren/research_data/simpler_env/rollouts/pizero_v1/bridge_beta_step19296_2024-12-26_22-30_42/widowx_carrot_on_plate/episode_96_success_True.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print(data[0]["sampled_action_embeds"].shape)
print(data[0]["img_embeds"].shape)

