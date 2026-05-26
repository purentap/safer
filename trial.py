import pickle
# path = f"/home/enes/puren/data/openvla_widowx/YEREL/openvla_widowx/put_the_carrot_on_plate_1/task_put_the_carrot_on_plate--ep9--succ0.pkl"
# with open(path, "rb") as f:
#     data = pickle.load(f)

# print(data["img_embeds"].shape)
# print(data["hidden_states"][0].shape)


# path2= "/home/enes/puren/data/openvla_data/openvla_data/single-forward/libero_10/task3--ep22--succ0.pkl"
# with open(path2, "rb") as f:
#     data2 = pickle.load(f)
# print(data2.keys())
# print(data2["hidden_states"].shape)

path = "/mnt/mahzen/rafi/vla-SAFE/puren_deneme/rollouts_all/20250509_rollouts/20250509_task4_put_the_cup_to_the_upright_position_02/env_records/task4--ep0--succ1--meta.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)
print(data.keys())
print(data["num_steps_wait"])
