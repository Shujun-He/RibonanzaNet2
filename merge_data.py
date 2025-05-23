import pickle
import numpy as np
import os

os.system("mkdir data")


# first merge data dicts
folder1 = "../test4/data"
folder2 = "../test18/data"

# RN1 data
with open(f"{folder1}/data_dict.p", "rb") as f:
    data_dict = pickle.load(f)
    # data_dict['SN']=data_dict['SN']#[:,[1,0]]
    # to_pad=len(data_dict['SN'])-len(data_dict['dataset_name'])
    # data_dict['dataset_name']=data_dict['dataset_name']+['NULL']*to_pad
    data_dict["source"] = ["Ribonanza2A"] * len(data_dict["SN"])
    data_dict["SN"] = np.pad(
        data_dict["SN"], ((0, 0), (0, 2)), "constant", constant_values=0
    )
    # exit()
    for key in data_dict.keys():
        print(key, len(data_dict[key]))


# RN2 data
with open(f"{folder2}/data_dict.p", "rb") as f:
    data_dict2 = pickle.load(f)
    data_dict2["source"] = ["Ribonanza2C"] * len(data_dict2["SN"])
    data_dict2["SN"] = np.pad(
        data_dict2["SN"], ((0, 0), (2, 0)), "constant", constant_values=0
    )
    print(data_dict2.keys())

#'sequences', 'SN', 'dataset_name'
new_data_dict = {}
# for key in data_dict2.keys():
#     new_data_dict[key]=list(data_dict[key])+list(data_dict2[key])

new_data_dict["sequences"] = data_dict["sequences"] + data_dict2["sequences"]
new_data_dict["SN"] = np.concatenate([data_dict["SN"], data_dict2["SN"]])
print(new_data_dict["SN"].shape)
new_data_dict["dataset_name"] = data_dict["dataset_name"] + data_dict2["dataset_name"]
new_data_dict["source"] = data_dict["source"] + data_dict2["source"]

# save small objects in a pickle file
with open("data/data_dict.p", "wb+") as f:
    pickle.dump(new_data_dict, f)
# exit()
# merge labels
shape_rn1 = np.load(f"{folder1}/data_shape.npy")
labels_rn1 = np.memmap(
    f"{folder1}/labels.mmap", dtype="float32", mode="r", shape=tuple(shape_rn1)
)

shape_rn2 = np.load(f"{folder2}/data_shape.npy")
labels_rn2 = np.memmap(
    f"{folder2}/labels.mmap", dtype="float32", mode="r", shape=tuple(shape_rn2)
)

seq_len = 177

new_shape = (shape_rn1[0] + shape_rn2[0], seq_len, 4)
mmap_array = np.memmap("data/labels.mmap", dtype="float32", mode="w+", shape=new_shape)
mmap_array[:] = np.nan

mmap_array[: shape_rn1[0], :, [0, 1]] = labels_rn1  # reverse the order of the labels
mmap_array[shape_rn1[0] :, :, [2, 3]] = labels_rn2
np.save("data/data_shape", new_shape)
