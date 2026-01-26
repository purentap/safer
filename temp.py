import pickle

path = "/mnt/mahzen/puren/research_data/pi0fast_droid/rollouts_all/20250510_rollouts/20250510_task10_close_the_drawer/env_records/task12--ep0--succ1--meta.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print(data["task_description"])
