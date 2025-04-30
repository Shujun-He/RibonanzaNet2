#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm
from utils import *
import os
from Diffusion import finetuned_RibonanzaNet


import sys

#sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
#from Network import finetuned_RibonanzaNet







#exit()



# In[2]:
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# # Config

# In[3]:



config=load_config_from_yaml("diffusion_config.yaml")



# Load data

train_sequences=pd.read_csv("/lustre/fs0/scratch/shujun/Stanford3D_dataprocessing_add_modified_nts3/train_sequences.v0.5.1.csv")
train_labels=pd.read_csv("/lustre/fs0/scratch/shujun/Stanford3D_dataprocessing_add_modified_nts3/train_solution.v0.5.1.csv")


# In[5]:


train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
train_labels["pdb_id"] 


# In[6]:


float('Nan')


# In[7]:


# all_xyz=[]

# for pdb_id in tqdm(train_sequences['target_id']):
#     df = train_labels[train_labels["pdb_id"]==pdb_id]
#     #break
#     xyz=df[['x_1','y_1','z_1']].to_numpy().astype('float32')
#     xyz[xyz<-1e17]=float('Nan');
#     all_xyz.append(xyz)


import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_one(pdb_id):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    xyz = df[['x_1', 'y_1', 'z_1']].to_numpy().astype('float32')
    xyz[xyz < -1e17] = float('nan')
    return xyz

# if __name__ == "__main__":
with Pool(processes=cpu_count()//8) as pool:
    all_xyz = list(tqdm(pool.imap(process_one, train_sequences['target_id']), total=len(train_sequences)))

#df


# In[8]:


# filter the data
# Filter and process data
filter_nan = []
max_len = 0
for xyz in all_xyz:
    if len(xyz) > max_len:
        max_len = len(xyz)

    #fill -1e18 masked sequences to nans
    
    #sugar_xyz = np.stack([nt_xyz['sugar_ring'] for nt_xyz in xyz], axis=0)
    filter_nan.append((np.isnan(xyz).mean() <= 0.5) & \
                      (len(xyz)<config.max_len_filter) & \
                      (len(xyz)>config.min_len_filter))

print(f"Longest sequence in train: {max_len}")

filter_nan = np.array(filter_nan)
non_nan_indices = np.arange(len(filter_nan))[filter_nan]

train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
all_xyz=[all_xyz[i] for i in non_nan_indices]


# In[9]:


#pack data into a dictionary

data={
      "sequence":train_sequences['sequence'].to_list(),
      "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
      "description": train_sequences['description'].to_list(),
      "all_sequences": train_sequences['all_sequences'].to_list(),
      "xyz": all_xyz
}

#save to pickle
with open("train_data.pkl", "wb+") as f:
    pickle.dump(data, f)

