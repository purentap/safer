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

path = "/home/enes/puren/data/pi0fast_data/pi0fast_data/rollouts/pi0fast-libero_10/policy_records/step_99--pi0fast-libero_10--task_0--ep_1--t_245--meta.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)
print(data.keys())
print(data["pre_logits"].shape)
print(data["base_rgb_patches_before_projection"].shape)