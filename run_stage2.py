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
    "max_len": 512,
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
diffusion = Diffusion(model,n_times=config['n_times'])#.cuda()

state_dict=torch.load("RibonanzaNet-3D-final-v2.pt")

#get rid of module. from ddp state dict
new_state_dict={}

for key in state_dict:
    new_state_dict[key[7:]]=state_dict[key]

model.load_state_dict(new_state_dict)

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


# In[14]:


import plotly.graph_objects as go
import numpy as np



# Example: Generate an Nx3 matrix
# xyz = train_dataset[200]['xyz']  # Replace this with your actual Nx3 data
# N = len(xyz)


# for _ in range(2): #plot twice because it doesnt show up on first try for some reason
#     # Extract columns
#     x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
#     # Create the 3D scatter plot
#     fig = go.Figure(data=[go.Scatter3d(
#         x=x, y=y, z=z,
#         mode='markers',
#         marker=dict(
#             size=5,
#             color=z,  # Coloring based on z-value
#             colorscale='Viridis',  # Choose a colorscale
#             opacity=0.8
#         )
#     )])
    
#     # Customize layout
#     fig.update_layout(
#         scene=dict(
#             xaxis_title="X",
#             yaxis_title="Y",
#             zaxis_title="Z"
#         ),
#         title="3D Scatter Plot"
#     )

# fig.show()
    


# In[15]:


train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)


# # Get RibonanzaNet¶
# We will add a linear layer to predict xyz of C1' atoms

# In[ ]:





# In[16]:





#model.decode(torch.ones(1,10).long().cuda(),torch.ones(1,10).long().cuda())


from tqdm import tqdm

epochs=20
cos_epoch=10


best_loss=np.inf
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=0.0001) #no weight decay following AF

batch_size=1

#for cycle in range(2):

criterion=torch.nn.CrossEntropyLoss(reduction='none')

#scaler = GradScaler()

logger=CSVLogger(["epoch","train_loss","val_loss","val_rmsd","val_lddt"],"log_stage2.csv")

schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs-cos_epoch)*len(train_loader)//batch_size)

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs 

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],mixed_precision='bf16')



model, optimizer, train_loader, val_loader, schedule, diffusion = accelerator.prepare(model, optimizer, train_loader, val_loader, schedule, diffusion)
#diffusion = accelerator.prepare(diffusion)

best_val_loss=99999999999
for epoch in range(epochs):
    model.train()
    tbar=tqdm(train_loader)
    total_loss=0
    total_distogram_loss=0
    oom=0
    for idx, batch in enumerate(tbar):
        #try:
        sequence=batch['sequence']#.cuda()
        gt_xyz=batch['xyz'].squeeze()

        gt_xyz[torch.isnan(gt_xyz)]=0

        distance_matrix=calculate_distance_matrix(gt_xyz,gt_xyz)
        distogram_mask=distance_matrix==distance_matrix
        distance_matrix=distance_matrix.clip(2,39).long()



        gt_xyz=gt_xyz.unsqueeze(0).repeat(48,1,1)
        time_steps=torch.randint(0,config['n_times'],size=(gt_xyz.shape[0],)).to(gt_xyz.device)
        if accelerator.distributed_type=='NO':
            noised_xyz, noise=diffusion.make_noisy(gt_xyz, time_steps)
        else:
            noised_xyz, noise=diffusion.module.make_noisy(gt_xyz, time_steps)

        #exit()
        with accelerator.autocast():
            pred_noise,distogram_pred=model(sequence,noised_xyz,time_steps)#.squeeze()
        #pred_xyz=aug_xyz[:,1:-1]+pred_displacements[:,1:-1]
        #exit()

        mask=~torch.isnan(noised_xyz)
        loss= torch.square(noise[mask]-pred_noise[mask]).mean()

        
        distogram_loss=criterion(distogram_pred.squeeze()[distogram_mask],distance_matrix[distogram_mask]).mean()
        total_distogram_loss+=distogram_loss.item()


        if loss!=loss:
            stop

        
        #(loss/batch_size*len(gt_xyz)).backward()

        accelerator.backward((loss+0.2*distogram_loss)/batch_size*len(gt_xyz))

        if (idx+1)%batch_size==0 or idx+1 == len(tbar):

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            # scaler.scale(loss/batch_size).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # scaler.step(optimizer)
            # scaler.update()

            
            if (epoch+1)>cos_epoch:
                schedule.step()
        #schedule.step()
        total_loss+=loss.item()
        
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)} Distogram Loss: {total_distogram_loss/(idx+1)}")
        #break
    # visualize_point_cloud_batch(pred_xyz)
    # visualize_point_cloud_batch(aug_xyz)

    total_loss=total_loss/len(tbar)

    tbar=tqdm(val_loader)
    model.eval()
    val_preds=[]
    val_loss=0
    val_rmsd=0
    val_lddt=0
    #unwrapped_diffusion=accelerator.unwrap_model(diffusion)
    #unwrapped_model=accelerator.unwrap_model(model)
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        gt_xyz=batch['xyz'].cuda().squeeze()

        with torch.no_grad():
            # if accelerator.dis
            #pred_xyz=model.module.decode(sequence,torch.ones_like(sequence).long().cuda()).squeeze()
            if accelerator.distributed_type=='NO':
                pred_xyz=diffusion.sample(sequence,1)[0].squeeze(0)
            else:
                pred_xyz=diffusion.module.sample(sequence,1)[0].squeeze(0)
            #pred_xyz=model(sequence)[-1].squeeze()
            loss=dRMAE(pred_xyz,pred_xyz,gt_xyz,gt_xyz)

        val_rmsd+=accelerator.gather(align_svd_rmsd(pred_xyz,gt_xyz)).mean().item()
        val_lddt+=accelerator.gather(torch.tensor(compute_lddt(pred_xyz.cpu().numpy(),gt_xyz.cpu().numpy())).to(pred_xyz.device)).mean().item()
        val_loss+=accelerator.gather(loss).mean().item()

        val_preds.append([gt_xyz.cpu().numpy(),pred_xyz.cpu().numpy()])
    val_loss=val_loss/len(tbar)
    val_rmsd=val_rmsd/len(tbar)
    val_lddt=val_lddt/len(tbar)

    print(f"val loss: {val_loss}")
    print(f"val_rmsd: {val_rmsd}")
    print(f"val_lddt: {val_lddt}")

    if accelerator.is_main_process:
        logger.log([epoch,total_loss,val_loss,val_rmsd,val_lddt])
        
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            best_preds=val_preds
            torch.save(model.state_dict(),'RibonanzaNet-3D-stage2-v2.pt')

    # 1.053595052265986 train loss after epoch 0
torch.save(model.state_dict(),'RibonanzaNet-3D-stage2-final-v2.pt')

