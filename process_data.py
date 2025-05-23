import polars as pl
from Dataset import *
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
import argparse
from accelerate import Accelerator
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import h5py
# from torch.cuda.amp import GradScaler
# from torch import autocast

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs/pairwise.yaml")

args = parser.parse_args()

np.random.seed(0)

config = load_config_from_yaml(args.config_path)


os.system("mkdir data")


data = h5py.File("../../Ribonanza2A_Genscript.v0.1.0.hdf5", "r")
sublib_data = pl.read_csv("../../sublib_id.csv")["sublibrary"].to_list()
# exit()
SN = data["signal_to_noise"][:]
sequences = data["sequences"][:]
labels = data["r_norm"][:]

sequences = [sequences[0, i].decode("utf-8") for i in range(sequences.shape[1])]

print("there are", len(sequences), "sequences")

data_dict = {
    "sequences": sequences,
    "labels": labels,
    "SN": SN,
}

# first get high quality data
# usable_snr_indices = SN.max(1)>=1
# high_quality_indices = SN.min(1)>=1


# save small objects in a pickle file
with open("data/data_dict.p", "wb+") as f:
    save_data_dict = {
        "sequences": data_dict["sequences"],
        "SN": data_dict["SN"],
        "dataset_name": sublib_data,
    }
    pickle.dump(save_data_dict, f)

# with open('data/dataset_name.p','wb+') as f:
#     pickle.dump(dataset_name,f)


# save labels
filename = "data/labels.mmap"  # Specify the path where the memmap file should be stored
dtype = "float32"  # Change this to match the dtype of your labels array
mode = "w+"  # Write mode, will create or overwrite existing file
shape = data_dict["labels"].shape  # Shape of the array

# Create the memmap array
mmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

mmap_array[:] = data_dict["labels"]
mmap_array.flush()

np.save("data/data_shape", shape)
