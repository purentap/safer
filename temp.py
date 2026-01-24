import pickle

path = "/mnt/mahzen/puren/research_data/simpler_env/rollouts/pizero_v1_remaining/fractal_beta_step29576_2024-12-29_13-10_42/google_robot_move_near_v0/episode_0_success_False.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print(data[0]["sampled_action_embeds"].shape)