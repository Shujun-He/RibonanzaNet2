import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from utils import *
import os
#from Diffusion import Diffusion

# config = {
#     "seed": 0,
#     "cutoff_date": "2020-01-01",
#     "test_cutoff_date": "2022-05-01",
#     "max_len": 384,
#     "batch_size": 1,
#     "learning_rate": 1e-4,
#     "weight_decay": 0.0,
#     "mixed_precision": "bf16",
#     "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
#     "epochs": 10,
#     "cos_epoch": 5,
#     "loss_power_scale": 1.0,
#     "max_cycles": 1,
#     "grad_clip": 0.1,
#     "gradient_accumulation_steps": 1,
#     "d_clamp": 30,
#     "max_len_filter": 9999999,
#     "structural_violation_epoch": 50,
#     "balance_weight": False,
# }

test_data=pd.read_csv("../CONFIDENTIAL/test_sequences_CONFIDENTIAL.csv")
solution=pd.read_csv("../CONFIDENTIAL/test_solution_CONFIDENTIAL.csv")

from torch.utils.data import Dataset, DataLoader

class RNADataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)




        return {'sequence':sequence}

test_dataset=RNADataset(test_data)



import sys

#sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
from Diffusion import finetuned_RibonanzaNet




config=load_config_from_yaml("diffusion_config.yaml")

model=finetuned_RibonanzaNet(load_config_from_yaml("pairwise.yaml"),config,pretrained=True).cuda()


#model.decode(torch.ones(1,10).long().cuda(),torch.ones(1,10).long().cuda())


import torch
state_dict=torch.load("RibonanzaNet-3D-final-v2.pt",map_location='cpu')

#get rid of module. from ddp state dict
new_state_dict={}

for key in state_dict:
    new_state_dict[key[7:]]=state_dict[key]

model.load_state_dict(new_state_dict)


os.system('mkdir casp16_distograms')

from tqdm import tqdm
model.eval()
preds=[]
for i in tqdm(range(len(test_dataset))):
    src=test_dataset[i]['sequence'].long()
    src=src.unsqueeze(0).cuda()
    target_id=test_data.loc[i,'target_id']
    target_solution=solution[solution['ID'].str.contains(target_id)]
    gt_xyz=target_solution[['x_1','y_1','z_1']].values
    gt_distogram=calculate_distance_matrix(torch.tensor(gt_xyz),torch.tensor(gt_xyz)).numpy().clip(2,39)
    #model.eval()

    #tmp=[]
    predicted_dm=[]
    #for _ in range(5):
    with torch.no_grad():
        xyz,distogram=model.sample(src,5)
        #xyz=xyz.squeeze()
    
    predicted_dm=[]
    for j in range(2):
        predicted_dm.append(calculate_distance_matrix(xyz[j],xyz[j]).cpu().numpy().clip(2,39))

        #tmp.append(xyz.cpu().numpy())
    

    plt.subplot(2,2,1)
    plt.imshow(gt_distogram,cmap='hot',interpolation='nearest')
    plt.title('ground truth distogram')
    #plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(distogram.cpu().numpy(),cmap='hot',interpolation='nearest')
    plt.title('predicted distogram')
    plt.subplot(2,2,3)
    plt.imshow(predicted_dm[0],cmap='hot',interpolation='nearest')
    plt.title('predicted structure distogram')
    plt.subplot(2,2,4)
    plt.imshow(predicted_dm[1],cmap='hot',interpolation='nearest')
    plt.title('predicted structure distogram')
    #plt.colorbar()
    plt.savefig(f"casp16_distograms/{target_id}.png")
    plt.clf()
    #exit()

    # model.eval()
    # with torch.no_grad():
    #     xyz=model(src)[-1].squeeze()
    #tmp.append(xyz.cpu().numpy())

    #tmp=np.stack(tmp,0)
    #exit()
    preds.append(xyz.cpu().numpy())


ID=[]
resname=[]
resid=[]
x=[]
y=[]
z=[]

data=[]

for i in range(len(test_data)):
    #print(test_data.loc[i])

    
    for j in range(len(test_data.loc[i,'sequence'])):
        # ID.append(test_data.loc[i,'sequence_id']+f"_{j+1}")
        # resname.append(test_data.loc[i,'sequence'][j])
        # resid.append(j+1) # 1 indexed
        row=[test_data.loc[i,'target_id']+f"_{j+1}",
             test_data.loc[i,'sequence'][j],
             j+1]

        for k in range(5):
            for kk in range(3):
                row.append(preds[i][k][j][kk])
        data.append(row)

columns=['ID','resname','resid']
for i in range(1,6):
    columns+=[f"x_{i}"]
    columns+=[f"y_{i}"]
    columns+=[f"z_{i}"]


submission=pd.DataFrame(data,columns=columns)


submission
submission.to_csv('submission_casp16.csv',index=False)