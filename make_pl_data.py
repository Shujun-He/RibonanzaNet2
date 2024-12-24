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
#from torch.cuda.amp import GradScaler
#from torch import autocast
import os

os.system('mkdir pl_data')

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")

args = parser.parse_args()

np.random.seed(0)

config = load_config_from_yaml(args.config_path)

pl_train=pl.read_parquet(f"{config.input_dir}/merged_noisy_train_test_3parts/top3_ensembled_unused_w_test5.parquet")
pl_test=pl.read_parquet(f"{config.input_dir}/merged_noisy_train_test_3parts/pl_preds_top3_w_ribonanzanet.parquet")

#fix exp type names
exp_test=pl_test['experiment_type'].to_list()

for i in range(len(exp_test)):
    if exp_test[i]=='2A3':
        exp_test[i]='2A3_MaP'
    elif exp_test[i]=='DMS':
        exp_test[i]='DMS_MaP'

pl_test=pl_test.with_columns(pl.Series(name="experiment_type", values=exp_test))
#exit()

test=pl.read_csv("../../input/test_sequences.csv")
pl_test=pl_test.join(test[['sequence','sequence_id']],how='left',on='sequence_id')


#zero fill nan cols in pl_train
for i in range(206,457+1):
    pl_train=pl_train.with_columns(pl.lit(0.0).alias(f"reactivity_{i:04d}"))
for i in range(1,457+1):
    pl_train=pl_train.with_columns(pl.lit(0.0).alias(f"reactivity_error_{i:04d}"))


pl_train=pl_train.with_columns(pl.lit(1).alias(f"SN_filter"))
#pl_train=pl_test.with_columns(pl.lit(1).alias(f"SN_filter"))

#

n=set(pl_test.columns)^set(pl_train.columns)
print(n&set(pl_test.columns))
print(n&set(pl_train.columns))

for c in ['dataset_name', 'sequence_right', 'reads']:
    pl_train=pl_train.drop(c)


n=set(pl_test.columns)^set(pl_train.columns)
print(n&set(pl_test.columns))
print(n&set(pl_train.columns))

pl_data=pl.concat([pl_train,pl_test],how='diagonal_relaxed')

pl_data=pl_data.with_columns(pl.lit(10.).alias(f"signal_to_noise"))

#exit()

data=pl_data



print("before dropping duplicates data shape is:",data.shape)
data=data.unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])
print("after dropping duplicates data shape is:",data.shape)
#data=data.sort(["signal_to_noise"],descending=True).unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])

n_sequences_total=len(data)//2
#get necessary data as lists and numpy arrays
seq_length=457



label_names=["reactivity_{:04d}".format(number+1) for number in range(seq_length)]
error_label_names=["reactivity_error_{:04d}".format(number+1) for number in range(seq_length)]

sequences=data.unique(subset=["sequence_id"],maintain_order=True)['sequence'].to_list()
sequence_ids=data.unique(subset=["sequence_id"],maintain_order=True)['sequence_id'].to_list()
labels=data[label_names].to_numpy().astype('float32').reshape(-1,2,seq_length).transpose(0,2,1)
errors=data[error_label_names].to_numpy().astype('float32').reshape(-1,2,seq_length).transpose(0,2,1)
SN=data['signal_to_noise'].to_numpy().astype('float32').reshape(-1,2)


assert len(sequences)==len(sequence_ids)==len(labels)==len(errors)==len(SN)

print("SN min:",SN.min())

data_dict = {
    'sequences': sequences,
    'sequence_ids': sequence_ids,
    'labels': labels,
    'errors': errors,
    'SN': SN,
}

#save small objects in a pickle file
with open('pl_data/data_dict.p','wb+') as f:
    save_data_dict = {
        'sequences': data_dict['sequences'],
        'sequence_ids': data_dict['sequence_ids'],
        'SN': data_dict['SN'],
    }
    pickle.dump(save_data_dict,f)


#save labels 
filename = 'pl_data/labels.mmap'  # Specify the path where the memmap file should be stored
dtype = 'float32'       # Change this to match the dtype of your labels array
mode = 'w+'             # Write mode, will create or overwrite existing file
shape = data_dict['labels'].shape  # Shape of the array

# Create the memmap array
mmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

mmap_array[:] = data_dict['labels']
mmap_array.flush()

#save errors 
filename = 'pl_data/errors.mmap'  # Specify the path where the memmap file should be stored
dtype = 'float32'       # Change this to match the dtype of your labels array
mode = 'w+'             # Write mode, will create or overwrite existing file
#shape = (167671, 206, 2)  # Shape of the array

# Create the memmap array
mmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

mmap_array[:] =  data_dict['errors']
mmap_array.flush()

print(f"data shape is {shape}")
np.save('pl_data/data_shape',shape)