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
from Diffusion import Diffusion
from scipy.stats import norm


import sys

#sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
from Network import finetuned_RibonanzaNet







#exit()



# In[2]:
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# # Config

# In[3]:


config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter": 10, 
    "structural_violation_epoch": 50,
    "balance_weight": False,
    'n_times': 1000
}

model=finetuned_RibonanzaNet(load_config_from_yaml("pairwise.yaml"),pretrained=True)#.cuda()
diffusion = Diffusion(model,n_times=200,beta_minmax=[0.01,0.07])#.cuda()

#pred=diffusion.sample(torch.ones(1,10).long().cuda(),5)

# # Get data and do some data processing¶
# 

# In[4]:


# Load data

train_sequences=pd.read_csv("../input/train_sequences.csv")
train_labels=pd.read_csv("../input/train_labels.csv")


# In[5]:


train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
train_labels["pdb_id"] 


# In[6]:


float('Nan')


# In[7]:


all_xyz=[]

for pdb_id in tqdm(train_sequences['target_id']):
    df = train_labels[train_labels["pdb_id"]==pdb_id]
    #break
    xyz=df[['x_1','y_1','z_1']].to_numpy().astype('float32')
    xyz[xyz<-1e17]=float('Nan');
    all_xyz.append(xyz)


df


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
                      (len(xyz)<config['max_len_filter']) & \
                      (len(xyz)>config['min_len_filter']))

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


# # Split train data into train/val/test¶
# We will simply do a temporal split, because that's how testing is done in structural biology in general (in actual blind tests)

# In[10]:


# Split data into train and test
all_index = np.arange(len(data['sequence']))
cutoff_date = pd.Timestamp(config['cutoff_date'])
test_cutoff_date = pd.Timestamp(config['test_cutoff_date'])
train_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) <= cutoff_date]
test_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) > cutoff_date and pd.Timestamp(d) <= test_cutoff_date]


# In[11]:


print(f"Train size: {len(train_index)}")
print(f"Test size: {len(test_index)}")
#exit()

# # Get pytorch dataset¶

# In[12]:


from torch.utils.data import Dataset, DataLoader
from ast import literal_eval

def get_ct(bp,s):
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0]-1,b[1]-1]=1
    return ct_matrix

from collections import defaultdict

class RNA3D_Dataset(Dataset):
    def __init__(self,indices,data):
        self.indices=indices
        self.data=data
        #set default to 4
        self.tokens=defaultdict(lambda: 4)
        self.tokens['A']=0
        self.tokens['C']=1
        self.tokens['G']=2
        self.tokens['U']=3

        #{nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        idx=self.indices[idx]
        sequence=[self.tokens[nt] for nt in (self.data['sequence'][idx])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        #get C1' xyz
        xyz=self.data['xyz'][idx]
        xyz=torch.tensor(np.array(xyz))


        if len(sequence)>config['max_len']:
            crop_start=np.random.randint(len(sequence)-config['max_len'])
            crop_end=crop_start+config['max_len']

            sequence=sequence[crop_start:crop_end]
            xyz=xyz[crop_start:crop_end]
        
        #center at first atom if first atom does not exit go until it does
        for i in range(len(xyz)):
            if (~torch.isnan(xyz[i])).all():
                break
        xyz=xyz-xyz[i]

        # for i in range(len(xyz)):

        #     if torch.isnan(xyz[i]).any():
        #         if i==0:
        #             xyz[i]=xyz[i+1]
        #         else:
        #             xyz[i]=xyz[i-1]

        return {'sequence':sequence,
                'xyz':xyz}


# In[13]:


train_dataset=RNA3D_Dataset(train_index,data)
val_dataset=RNA3D_Dataset(test_index,data)

train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)

tbar=tqdm(train_loader)

all_xyz=[]
for idx, batch in enumerate(tbar):
    #try:
    sequence=batch['sequence']#.cuda()
    gt_xyz=batch['xyz'].squeeze()

    gt_xyz[torch.isnan(gt_xyz)]=0

    gt_xyz=gt_xyz.unsqueeze(0).repeat(2,1,1)
    time_steps=torch.randint(0,config['n_times'],size=(gt_xyz.shape[0],))
    time_steps[:]=199
    noised_xyz, noise=diffusion.make_noisy(gt_xyz, time_steps)
    all_xyz.append(noised_xyz.reshape(-1).numpy())
    #exit()

all_xyz=np.concat(all_xyz,axis=0)

# Create histogram of the data, normalized to density
counts, bins, _ = plt.hist(all_xyz, bins=100, density=True, alpha=0.6, color='blue', label='Histogram')

# Generate x values over the range of your data
x = np.linspace(min(bins), max(bins), 500)

# Standard normal PDF (mean=0, std=1)
pdf = norm.pdf(x, loc=0, scale=1)

# Plot standard normal PDF
plt.plot(x, pdf, color='red', label='Standard Normal PDF')

# Labels and title
plt.grid()
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram of all xyz values with Standard Normal PDF')
plt.legend()

# Save and show
plt.savefig('all_xyz_final_step_histogram_with_std_normal_pdf.png')
plt.show()
