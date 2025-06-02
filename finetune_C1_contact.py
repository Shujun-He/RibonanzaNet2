#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import argparse
import yaml
import os


os.system('mkdir c1_contact_grid_search_scores')
os.system('mkdir c1_contact_grid_search_results')

parser = argparse.ArgumentParser(description="Train model with YAML configuration file.")

# Add argument for YAML configuration file
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the YAML configuration file."
)
parser.add_argument(
    "--max_len",
    type=int,
    default=1024,
    help="filter sequences longer than this length"
)
parser.add_argument(
    "--debug",
    action='store_true',
    help="Run in debug mode with reduced dataset size."
)

args = parser.parse_args()

with open(args.config, "r") as file:
    finetune_config = yaml.safe_load(file)
# In[2]:


train_data=pd.read_pickle("../../input/C1_contact/train_sequences_clustered.pkl")
val_data=pd.read_pickle("../../input/C1_contact/validation_sequences_clustered.pkl")

train_data['length']=train_data['sequence'].apply(len)
val_data['length']=val_data['sequence'].apply(len)

train_data=train_data.loc[train_data['length']<=int(args.max_len)].reset_index(drop=True)
val_data=val_data.loc[val_data['length']<=int(args.max_len)].reset_index(drop=True)
# In[12]:

train_data=train_data.drop_duplicates('cluster').reset_index(drop=True)
val_data=val_data.drop_duplicates('cluster').reset_index(drop=True)

print(f"Train data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")


from torch.utils.data import Dataset, DataLoader
import pickle
from torch.nn.functional import one_hot 

def get_ct(bp,s):
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0]-1,b[1]-1]=1
    return ct_matrix

def get_c1_concat_map(xyz, cutoff=15.0):
    """
    Convert a 3D coordinate tensor to a C1 concatenation map.
    """
    # Assuming xyz is of shape (batch_size, num_atoms, 3)
    num_atoms, _ = xyz.shape
    #c1_concat_map = torch.zeros((num_atoms, num_atoms), dtype=torch.float32)
    c1_distance= np.square(xyz[:,None]- xyz[None,:]).sum(-1)**0.5
    c1_concat_map = (c1_distance < cutoff).astype(np.float32)
    #mask the diagonal and close contacts

    for i in range(num_atoms):
        for j in range(num_atoms):
            if abs(i - j) < 4:  # Masking close contacts
                c1_concat_map[i, j] = 0.0
    return c1_concat_map

from collections import defaultdict

class RNA2D_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens=defaultdict(lambda: 4) #default to 4 for unknown nucleotides
        self.tokens.update({'A':0, 'C':1, 'G':2, 'U':3})
        #self.tokens={nt:i for i,nt in enumerate('ACGU')}
        #self.msa_data=pickle.load(open('../trrosetta_msa_data.p','rb'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=self.data.loc[idx,'sequence']

        sequence=[self.tokens[nt] for nt in sequence]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        ct=get_c1_concat_map(self.data.loc[idx,'xyz'])

        ct=torch.tensor(ct)

        

        return {'sequence':sequence,
                'ct':ct,}
                #'msa':msa}


# In[23]:


train_dataset=RNA2D_Dataset(train_data)
val_dataset=RNA2D_Dataset(val_data)
#test_dataset=RNA2D_Dataset(test_data)

#exit()

# In[13]:


train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)


plt.imshow(train_dataset[1]['ct'].numpy())
plt.title("C1 contact map")
plt.colorbar()
plt.savefig("c1_contact.png")
exit()

# In[14]:


from Network import *
import yaml

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

pretrain_logs=pd.read_csv('logs/fold0.csv')
best_epoch=pretrain_logs['val_loss'].argmin()
#best_epoch=9
best_weights_path=f"models/epoch_{best_epoch}/pytorch_model_fsdp.bin"
#best_weights_path='models/step_22000/pytorch_model_fsdp.bin'

print(f"best_weights_path: {best_weights_path}")

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout=0.1
        config.use_grad_checkpoint=True
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load(best_weights_path,map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)



        self.ct_predictor = nn.Sequential(
                                            nn.LayerNorm(config.pairwise_dimension),  # LayerNorm applied first
                                            nn.Linear(config.pairwise_dimension, 1)   # Linear layer with output dimension of 1
                                         )

    def forward(self,src):
        
        #with torch.no_grad():
        _, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        pairwise_features=pairwise_features+pairwise_features.permute(0,2,1,3)

        # msa_pair_freq=torch.einsum('bnic,bnjd->bijcd',msa,msa)
        # msa_pair_freq = rearrange(msa_pair_freq, 'b i j c d -> b i j (c d)')
        # msa_pair_freq=self.msa_embedding(msa_pair_freq)

        # features=torch.cat([pairwise_features,msa_pair_freq],-1)
        # features=self.MLP(features)

        output=self.ct_predictor(self.dropout(pairwise_features))

        #output=output+output.permute()

        return output.squeeze(-1)


# In[15]:


from sklearn.metrics import f1_score
import os

# Set the environment variable
os.environ['ARNIEFILE'] = '../arnie_file.txt'

# To check if the variable was set correctly, you can print it
print(os.environ['ARNIEFILE'])

from arnie.pk_predictors import _hungarian
from arnie.utils import convert_dotbracket_to_bp_list

def mask_diagonal(matrix, mask_value=0):
    matrix=matrix.copy()
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 3:
                matrix[i][j] = mask_value
    return matrix

def tune_val_f1(val_preds):
    THs=np.linspace(0,0.9,10)
    best_f1=0
    best_theta=0
    best_f1_list=[]
    for th in THs:
        
        F1s=[]
        
        for y_true, y_pred in val_preds:
            # s,bp=_hungarian(mask_diagonal(y_pred[0]),theta=th,min_len_helix=1)
            # ct_matrix=np.zeros((len(s),len(s)))
            # for b in bp:
            #     ct_matrix[b[0],b[1]]=1
            # ct_matrix=ct_matrix+ct_matrix.T
            ct_matrix = y_pred[0] > th
            F1s.append(f1_score(y_true.reshape(-1),ct_matrix.reshape(-1)))
        # print(f"Using TH {th}, avg F1 score is")    
        # print(np.mean(F1s))
        if np.mean(F1s)>best_f1:
            best_f1_list=F1s
            best_f1=np.mean(F1s)
            best_theta=th

    print(f"Using TH {best_theta}, avg F1 score is")    
    print(best_f1)  
    return best_f1, best_theta


# In[16]:


from tqdm import tqdm
model=finetuned_RibonanzaNet(load_config_from_yaml("configs/pairwise.yaml"),pretrained=True).cuda()
#https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb

# Load hyperparameters from the finetune_config dict
epochs = finetune_config["epochs"]
cos_epoch = finetune_config["cos_epoch"]
loss_power_scale = finetune_config["loss_power_scale"]
upweight_positive = finetune_config["upweight_positive"]

normalize_length = train_data['length'].mean() ** loss_power_scale

best_loss = float('inf')
optimizer = torch.optim.Adam(
    model.parameters(), 
    weight_decay=finetune_config["optimizer_weight_decay"], 
    lr=finetune_config["optimizer_lr"]
)

batch_size = finetune_config["batch_size"]

criterion = torch.nn.BCEWithLogitsLoss(reduction=finetune_config["criterion_reduction"])

schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=epochs * len(train_loader) * 0.25 // batch_size
)

cos_steps = epochs * len(train_loader) * 0.75

prefix=args.config.split('/')[-1]

os.system(f'mkdir c1_contact_grid_search_weights')
best_f1 = 0
total_steps = 0
for epoch in range(epochs):
    model.train()
    tbar = tqdm(train_loader)
    total_loss = 0
    oom = 0

    for idx, batch in enumerate(tbar):
        total_steps+=1
        try:
            sequence = batch['sequence'].cuda()
            labels = batch['ct'].cuda()

            output = model(sequence)

            # Apply loss and weighting
            loss = criterion(output, labels)
            loss[labels == 1] = loss[labels == 1] * upweight_positive
            loss = loss.mean() * sequence.shape[1] ** loss_power_scale / normalize_length

            # Backward pass
            (loss / batch_size).backward()

            if (idx + 1) % batch_size == 0 or idx + 1 == len(tbar):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    finetune_config["gradient_clip"]
                )
                optimizer.step()
                optimizer.zero_grad()

                if total_steps > cos_steps:
                    schedule.step()

            total_loss += loss.item()
            tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss / (idx + 1)} OOMs: {oom}")
            if args.debug:
                break
            #break
        except Exception:
            oom += 1

    # Validation loop
    tbar = tqdm(val_loader)
    model.eval()
    val_preds = []
    val_loss = 0

    for idx, batch in enumerate(tbar):
        sequence = batch['sequence'].cuda()
        labels = batch['ct'].cuda()

        with torch.no_grad():
            output = model(sequence)
            loss = criterion(output, labels)
            loss = loss.mean()

        val_loss += loss.item()
        val_preds.append([labels.cpu().numpy(), output.sigmoid().cpu().numpy()])
        
        if args.debug:
            break
    print(f"Validation loss: {val_loss}")

    f1, th = tune_val_f1(val_preds)

    if f1 > best_f1:
        best_f1 = f1
        best_theta = th
        best_preds = val_preds
        #torch.save(model.state_dict(), finetune_config["checkpoint_path"])
        best_state_dict=model.state_dict().copy()
        torch.save(best_state_dict, f"c1_contact_grid_search_weights/{prefix}_best_weights.pth")
        
        with open(f"c1_contact_grid_search_scores/{prefix}.txt",'w+') as f:
            f.write(str(best_f1))

    
# In[17]:


model.load_state_dict(best_state_dict)
model.eval()

#compute test set F1 scores
for year in range(2019, 2025):
    test_data = pd.read_pickle(f"../../input/C1_contact/test_sequences_clustered_{year}.pkl")

    #impose max length filter
    test_data['length'] = test_data['sequence'].apply(len)
    test_data = test_data.loc[test_data['length'] <= int(args.max_len)].reset_index(drop=True)

    print(f"Test data length for {year}: {len(test_data)}")

    test_dataset=RNA2D_Dataset(test_data)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)

    F1s = []
    for batch in tqdm(test_loader):
        sequence = batch['sequence'].cuda()
        true_contact = batch['ct'].numpy().squeeze()

        with torch.no_grad():
            output = model(sequence).sigmoid()
            predicted_contact = output.cpu().numpy() > best_theta
        

        val_preds.append(predicted_contact)
        F1s.append(f1_score(true_contact.reshape(-1), predicted_contact.reshape(-1)))

    test_data['f1_score'] = F1s
    test_data.to_pickle(f"c1_contact_grid_search_results/{prefix}_test_f1_scores_{year}.pkl")
    print(f"Test F1 scores for {year}: {np.mean(F1s)}")

    #compute F1 score for the test set
