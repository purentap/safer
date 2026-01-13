import pickle
path = "/mnt/mahzen/puren/research_data/pi0_data/pi0_safe_data/rollouts/pi0-libero_10/policy_records/step_9999--pi0-libero_10--task_3--ep_15--t_175--meta.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print(data.keys())

