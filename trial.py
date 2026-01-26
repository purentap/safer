import pickle
path = f"/mnt/mahzen/puren/research_data/simpler_env/rollouts/pizero_v1/fractal_beta_step29576_2024-12-29_13-10_42/google_robot_close_drawer/episode_68_success_True.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print(data[0]["sampled_action_embeds"].shape)
print(data[0]["img_embeds"].shape)

