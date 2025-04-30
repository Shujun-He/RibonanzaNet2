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

# In[2]:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  


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
}


# # Get data and do some data processing¶
# 

# In[4]:


# Load data

train_sequences=pd.read_csv("train_sequences.csv")
train_labels=pd.read_csv("train_labels.csv")


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


# # Get pytorch dataset¶

# In[12]:


from torch.utils.data import Dataset, DataLoader
from ast import literal_eval

def get_ct(bp,s):
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0]-1,b[1]-1]=1
    return ct_matrix

class RNA3D_Dataset(Dataset):
    def __init__(self,indices,data):
        self.indices=indices
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

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


import sys

sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
from Network import RibonanzaNet, MultiHeadAttention
import yaml
from CausalDecoder import SimpleStructureModule



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)



class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout=0.1
        config.use_grad_checkpoint=True
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load("RibonanzaNet.pt",map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)

        decoder_dim=768
        self.structure_module=[SimpleStructureModule(d_model=decoder_dim, nhead=8, 
                 dim_feedforward=decoder_dim*4, pairwise_dimension=64, dropout=0.1) for i in range(6)]
        self.structure_module=nn.ModuleList(self.structure_module)

        self.xyz_embedder=nn.Linear(3,decoder_dim)
        self.xyz_norm=nn.LayerNorm(decoder_dim)
        self.xyz_predictor=nn.Linear(decoder_dim,3)
        
        self.adaptor=nn.Sequential(nn.Linear(256,decoder_dim),nn.LayerNorm(decoder_dim))

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward
    
    def forward(self,src,xyz):
        
        #with torch.no_grad():
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        
        sequence_features=self.adaptor(sequence_features)

        decoder_batch_size=xyz.shape[0]
        sequence_features=sequence_features.repeat(decoder_batch_size,1,1)
        tgt=self.xyz_norm(sequence_features+self.xyz_embedder(xyz))
        # print(tgt)
        # exit()
        pairwise_features=pairwise_features.expand(decoder_batch_size,-1,-1,-1)
        # print(tgt.shape)
        # print(sequence_features.shape)
        # print(pairwise_features.shape)
        # exit()        


        # print(tgt.shape)
        # exit()
        # tgt=tgt.repeat(decoder_batch_size,1,1)
        # sequence_features=sequence_features.repeat(decoder_batch_size,1,1)
        # pairwise_features=pairwise_features.expand(decoder_batch_size,-1,-1,-1)
        # print(xyz.shape)
        # exit()


        for layer in self.structure_module:
            tgt=layer([tgt, sequence_features,pairwise_features,xyz,None])
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        
        xyz=self.xyz_predictor(tgt).squeeze(0)
        # print(xyz.shape)
        # exit()
        return xyz
    

    def decode(self,src,src_mask):
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        sequence_features=self.adaptor(sequence_features)

        init_xyz=torch.zeros(1,2,3).to(src.device)
        init_xyz[:,1,0]=5.4

        for i in range(2,src.shape[1]):
            tgt=sequence_features[:,:i]
            tgt=self.xyz_norm(tgt+self.xyz_embedder(init_xyz))
            #pairwise_features=pairwise_features[:,:i,:i]
            #mask=torch.ones_like(src_mask)
            #print(tgt.shape)
            for layer in self.structure_module:
                tgt=layer([tgt, sequence_features,pairwise_features[:,:i,:i],init_xyz,src_mask[:,:i]])
            pred_displacements=self.xyz_predictor(tgt[:,-1])
            next_xyz=init_xyz[:,-1]+pred_displacements
            # print(next_xyz.shape)
            # print(init_xyz.shape)
            # exit()
            init_xyz=torch.cat([init_xyz,next_xyz[:,None,:]],1)
            # print(init_xyz.shape)
            # exit()
        return init_xyz


def random_rotation_point_cloud_torch(point_cloud):
    """
    Apply a random 3D rotation to a Nx3 point cloud (PyTorch version).
    
    Args:
        point_cloud (torch.Tensor): Nx3 tensor of XYZ points.

    Returns:
        torch.Tensor: Rotated Nx3 point cloud.
    """
    # Generate a random 3x3 matrix
    A = torch.randn(3, 3).to(point_cloud.device)
    # QR decomposition to get an orthonormal matrix Q
    Q, R = torch.linalg.qr(A)
    # Ensure it's a proper rotation matrix (det = +1)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    point_cloud = point_cloud + ((torch.rand(3)-0.5)*20)[None,:].to(point_cloud.device)
    rotated = point_cloud @ Q.T
    #add random rotation as well
    

    return rotated

model=finetuned_RibonanzaNet(load_config_from_yaml("pairwise.yaml"),pretrained=True).cuda()

model.load_state_dict(torch.load("RibonanzaNet-3D-final-v2.pt"))

#model(torch.ones(1,10).long().cuda())

model.decode(torch.ones(1,10).long().cuda(),torch.ones(1,10).long().cuda())
# exit()
# In[ ]:





# In[ ]:





# # Training loop¶
# we will use dRMSD loss on the predicted xyz. the loss function is invariant to translations, rotations, and reflections. because dRMSD is invariant to reflections, it cannot distinguish chiral structures, so there may be better loss functions

# In[17]:


def calculate_distance_matrix(X,Y,epsilon=1e-4):
    return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()


def dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    if d_clamp is not None:
        rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
    else:
        rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

    return rmsd.sqrt().mean()/Z


def dMSE(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    if d_clamp is not None:
        mse=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
    else:
        mse=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

    return mse.mean()


def local_dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=30):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=(~torch.isnan(gt_dm))*(gt_dm<d_clamp)
    mask[torch.eye(mask.shape[0]).bool()]=False



    rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon
    # rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).sqrt()/Z
    #rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])/Z
    #return rmsd.sqrt().mean()/Z
    return rmsd.mean()

def dRMAE(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])

    return rmsd.mean()/Z

import torch

def align_svd_mae(input, target, Z=10):
    """
    Aligns the input (Nx3) to target (Nx3) using SVD-based Procrustes alignment
    and computes RMSD loss.
    
    Args:
        input (torch.Tensor): Nx3 tensor representing the input points.
        target (torch.Tensor): Nx3 tensor representing the target points.
    
    Returns:
        aligned_input (torch.Tensor): Nx3 aligned input.
        rmsd_loss (torch.Tensor): RMSD loss.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    #mask 
    mask=~torch.isnan(target.sum(-1))

    input=input[mask]
    target=target[mask]
    
    # Compute centroids
    centroid_input = input.mean(dim=0, keepdim=True)
    centroid_target = target.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input - centroid_input.detach()
    target_centered = target - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)

    # Compute rotation matrix
    R = Vt @ U.T

    # Ensure a proper rotation (det(R) = 1, no reflection)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T

    # Rotate input
    aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()

    # # Compute RMSD loss
    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())

    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())
    
    # return aligned_input, rmsd_loss
    return torch.abs(aligned_input-target).mean()/Z


def align_svd_rmsd(input, target):
    """
    Aligns the input (Nx3) to target (Nx3) using SVD-based Procrustes alignment
    and computes RMSD loss.
    
    Args:
        input (torch.Tensor): Nx3 tensor representing the input points.
        target (torch.Tensor): Nx3 tensor representing the target points.
    
    Returns:
        aligned_input (torch.Tensor): Nx3 aligned input.
        rmsd_loss (torch.Tensor): RMSD loss.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    #mask 
    mask=~torch.isnan(target.sum(-1))

    input=input[mask]
    target=target[mask]
    
    # Compute centroids
    centroid_input = input.mean(dim=0, keepdim=True)
    centroid_target = target.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input - centroid_input.detach()
    target_centered = target - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)

    # Compute rotation matrix
    R = Vt @ U.T

    # Ensure a proper rotation (det(R) = 1, no reflection)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T

    # Rotate input
    aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()

    # # Compute RMSD loss
    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())

    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())
    
    # return aligned_input, rmsd_loss
    return torch.square(aligned_input-target).mean().sqrt()

# In[18]:


#pred_xyz=model(sequence)


# In[ ]:





# In[19]:


from tqdm import tqdm
#from torch.amp import GradScaler

epochs=10
cos_epoch=5


best_loss=np.inf
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=0.0001) #no weight decay following AF

batch_size=1

#for cycle in range(2):

criterion=torch.nn.BCEWithLogitsLoss(reduction='none')

#scaler = GradScaler()

logger=CSVLogger(["epoch","train_loss","val_loss","val_rmsd","val_lddt"],"log.csv")

schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs-cos_epoch)*len(train_loader)//batch_size)

best_val_loss=99999999999
for epoch in range(epochs):
    model.train()
    tbar=tqdm(train_loader)
    total_loss=0
    oom=0
    for idx, batch in enumerate(tbar):
        #try:
        sequence=batch['sequence'].cuda()
        gt_xyz=batch['xyz'].cuda().squeeze()

        gt_xyz[torch.isnan(gt_xyz)]=0

        aug_xyz=[]
        for _ in range(16):
            rotated_xyz=random_rotation_point_cloud_torch(gt_xyz)
            aug_xyz.append(rotated_xyz)
        aug_xyz=torch.stack(aug_xyz,0)

        #with torch.amp.autocast():
        pred_displacements=model(sequence,aug_xyz)#.squeeze()
        pred_xyz=aug_xyz[:,1:-1]+pred_displacements[:,1:-1]
        #exit()

        mask=(~torch.isnan(aug_xyz[:,2:]))*(~torch.isnan(pred_xyz))
        loss= torch.square(pred_xyz[mask]-aug_xyz[:,2:][mask]).mean()



        
        (loss/batch_size*len(gt_xyz)).backward()

        if (idx+1)%batch_size==0 or idx+1 == len(tbar):

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
        
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)} OOMs: {oom}")
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
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        gt_xyz=batch['xyz'].cuda().squeeze()

        with torch.no_grad():
            pred_xyz=model.decode(sequence,torch.ones_like(sequence).long().cuda()).squeeze()
            #pred_xyz=model(sequence)[-1].squeeze()
            loss=dRMAE(pred_xyz,pred_xyz,gt_xyz,gt_xyz)
            val_rmsd+=align_svd_rmsd(pred_xyz,gt_xyz)
            val_lddt+=compute_lddt(pred_xyz.cpu().numpy(),gt_xyz.cpu().numpy())


        val_loss+=loss.item()
        val_preds.append([gt_xyz.cpu().numpy(),pred_xyz.cpu().numpy()])
    val_loss=val_loss/len(tbar)
    val_rmsd=val_rmsd/len(tbar)
    val_lddt=val_lddt/len(tbar)

    print(f"val loss: {val_loss}")
    print(f"val_rmsd: {val_rmsd}")
    print(f"val_lddt: {val_lddt}")

    logger.log([epoch,total_loss,val_loss,val_rmsd,val_lddt])
    
    if val_loss<best_val_loss:
        best_val_loss=val_loss
        best_preds=val_preds
        torch.save(model.state_dict(),'RibonanzaNet-3D-finetuned.pt')

    # 1.053595052265986 train loss after epoch 0
torch.save(model.state_dict(),'RibonanzaNet-3D-final-finetuned.pt')

