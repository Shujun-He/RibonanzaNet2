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
from sklearn.metrics import f1_score
import os
from accelerate import Accelerator

# Set the environment variable
os.environ['ARNIEFILE'] = '../arnie_file.txt'

# To check if the variable was set correctly, you can print it
print(os.environ['ARNIEFILE'])

from arnie.pk_predictors import _hungarian
from arnie.utils import convert_dotbracket_to_bp_list

os.system('mkdir grid_search_scores')

parser = argparse.ArgumentParser(description="Train model with YAML configuration file.")

# Add argument for YAML configuration file
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the YAML configuration file."
)

args = parser.parse_args()

with open(args.config, "r") as file:
    finetune_config = yaml.safe_load(file)
# In[2]:


data=pd.read_parquet("../david_NAKB_dec/pdb_ss_data_w_pub_dates.parquet")
# #data=data.loc[data['length']<400].reset_index(drop=True)
data=data.drop_duplicates('dbn')
data=data.drop_duplicates('RNA_sequence')
data['publication_date']=[pd.Timestamp(s) for s in data['publication_date']]
data.shape


bp_rna_data=pd.read_csv("../../bpRNA/compiled.csv")
bp_rna_data['len']=bp_rna_data['sequence'].apply(lambda x: len(x))
bp_rna_data=bp_rna_data.loc[bp_rna_data['len']<1024].reset_index(drop=True)
#bp_rna_data['pairs']=[convert_dotbracket_to_bp_list(s, allow_pseudoknots=True) for s in bp_rna_data['structure']]

#exit()

# Define the cutoff date
atom1_cutoff = pd.Timestamp('2020-05-01')

# Filter rows where 'publication_date' is on or after the cutoff date
test_data = data[data['publication_date'] >= atom1_cutoff].reset_index(drop=True)
test_data.shape


# In[8]:


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# In[18]:


filter_no_structure=[]
for id,dbn in zip(data['Sequence_ID'],data['dbn']):
    if set(list(dbn))==set('.'):
        filter_no_structure.append(False)
    else:
        filter_no_structure.append(True)


# 

# In[19]:


data=data.loc[filter_no_structure].reset_index(drop=True)
data.shape


# In[9]:


train_data=data.loc[data['train_or_test']=='train'].reset_index(drop=True)
test_data=data.loc[data['train_or_test']=='test'].reset_index(drop=True)
test_data.shape


# In[11]:


#use 2 years data as val
val_cutoff = pd.Timestamp('2018-05-01')

train_split = train_data[train_data['publication_date'] < val_cutoff].reset_index(drop=True)
val_split = train_data[train_data['publication_date'] >= val_cutoff].reset_index(drop=True)

print(train_split.shape)
print(val_split.shape)


# In[12]:


from torch.utils.data import Dataset, DataLoader
import pickle
from torch.nn.functional import one_hot 

def get_ct(bp,s):
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0]-1,b[1]-1]=1
    return ct_matrix

def get_ct2(bp,s):
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0],b[1]]=ct_matrix[b[1],b[0]]=1
    return ct_matrix

class RNA2D_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        #self.msa_data=pickle.load(open('../trrosetta_msa_data.p','rb'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=self.data.loc[idx,'RNA_sequence']
        # if sequence in self.msa_data:
        #     msa=one_hot(torch.tensor(self.msa_data[sequence]).long(),5)
        # else:
        #     msa=one_hot(torch.ones(1,len(sequence)).long()*4,5)
        

        sequence=[self.tokens[nt] for nt in sequence]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        ct=get_ct(self.data.loc[idx,'pairs'],sequence)

        ct=torch.tensor(ct)

        

        return {'sequence':sequence,
                'ct':ct,}
                #'msa':msa}

from collections import defaultdict

class bpRNA2D_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens=defaultdict(lambda: 4)
        
        for i,nt in enumerate('ACGUN'):
            self.tokens[nt]=i
        #self.msa_data=pickle.load(open('../trrosetta_msa_data.p','rb'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=self.data.loc[idx,'sequence'].upper()
        sequence=[self.tokens[nt] for nt in sequence]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)


        dbn=self.data.loc[idx,'structure']
        pairs=convert_dotbracket_to_bp_list(dbn, allow_pseudoknots=True)
        #print(pairs)
        ct=get_ct2(pairs,sequence)

        ct=torch.tensor(ct)

        

        return {'sequence':sequence,
                'ct':ct,}
                #'msa':msa}




# In[23]:

bp_dataset=bpRNA2D_Dataset(bp_rna_data)
train_dataset=RNA2D_Dataset(train_split)
val_dataset=RNA2D_Dataset(val_split)
test_dataset=RNA2D_Dataset(test_data)

# plt.imshow(bp_dataset[0]['ct'].numpy(),cmap='gray')
# plt.savefig('bpRNA_example.png')
# exit()

# In[13]:

bp_loader=DataLoader(bp_dataset,batch_size=1,shuffle=True)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)


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
        config.dropout=0.2
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
            s,bp=_hungarian(mask_diagonal(y_pred[0]),theta=th,min_len_helix=1)
            ct_matrix=np.zeros((len(s),len(s)))
            for b in bp:
                ct_matrix[b[0],b[1]]=1
            ct_matrix=ct_matrix+ct_matrix.T
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
    T_max= (epochs - 2) * len(train_loader) // batch_size
)

cos_epoch = epochs - 3 

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs 

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],mixed_precision='bf16')

model, optimizer, train_loader, val_loader, bp_loader, schedule = \
accelerator.prepare(model, optimizer, train_loader, val_loader, bp_loader, schedule)

best_f1 = 0
total_steps = 0
best_val_loss = 999999
for epoch in range(epochs):
    model.train()
    
    total_loss = 0
    oom = 0
    if epoch <= cos_epoch: #train with bprna for one epoch
        tbar = tqdm(bp_loader)
    else:
        tbar = tqdm(train_loader)
    for idx, batch in enumerate(tbar):
        total_steps+=1
        #try:
        sequence = batch['sequence']#.cuda()
        labels = batch['ct']#.cuda()

        with accelerator.autocast():
            output = model(sequence)

            # Apply loss and weighting
            loss = criterion(output, labels)
            loss[labels == 1] = loss[labels == 1] * upweight_positive
            loss = loss.mean() * sequence.shape[1] ** loss_power_scale / normalize_length

        # Backward pass
        (loss / batch_size).backward()

        if (idx + 1) % batch_size == 0 or idx + 1 == len(tbar):
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), 
            #     finetune_config["gradient_clip"]
            # )
            # optimizer.step()
            # optimizer.zero_grad()
            accelerator.clip_grad_norm_(model.parameters(), finetune_config["gradient_clip"])
            optimizer.step()
            #optimizer_step()
            optimizer.zero_grad()

            if epoch > cos_epoch:
                schedule.step()

        total_loss += loss.item()
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss / (idx + 1)} OOMs: {oom}")
        #break
        # except Exception:
        #     oom += 1

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

        #val_loss += loss.item()

        val_loss += accelerator.gather(loss).mean().item()

        #val_preds.append([labels.cpu().numpy(), output.sigmoid().cpu().numpy()])
    
    val_loss=val_loss / len(val_loader)
    print(f"Validation loss: {val_loss}")



    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict=model.state_dict().copy()
        torch.save(best_state_dict, f"bp_RNA_best_model.pth")


best_theta = 0.5


model.load_state_dict(best_state_dict)
model.eval()

test_preds=[]
test_labels=[]

for i in tqdm(range(len(test_dataset))):
    example=test_dataset[i]
    sequence=example['sequence'].cuda().unsqueeze(0)
    labels=example['ct'].cuda()

    with torch.no_grad():
        test_preds.append(model(sequence).sigmoid().cpu().numpy())
        test_labels.append(labels.cpu().numpy())


# In[18]:


test_preds_hungarian=[]
hungarian_structures=[]
for i in range(len(test_preds)):
    s,bp=_hungarian(mask_diagonal(test_preds[i][0]),theta=best_theta,min_len_helix=1)
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0],b[1]]=1
    ct_matrix=ct_matrix+ct_matrix.T
    test_preds_hungarian.append(ct_matrix)
    hungarian_structures.append(s)


# In[19]:


test_hungarian_F1s=[]
for y_true, y_pred in zip(test_labels, test_preds_hungarian):
    test_hungarian_F1s.append(f1_score(y_true.reshape(-1),y_pred.reshape(-1)))
np.mean(test_hungarian_F1s)


# In[20]:


def detect_crossed_pairs(bp_list):
    """
    Detect crossed base pairs in a list of base pairs in RNA secondary structure.

    Args:
    bp_list (list of tuples): List of base pairs, where each tuple (i, j) represents a base pair.
    
    Returns:
    list of tuples: List of crossed base pairs.
    """
    crossed_pairs_set = set()
    crossed_pairs = []
    # Iterate through each pair of base pairs
    for i in range(len(bp_list)):
        for j in range(i+1, len(bp_list)):
            bp1 = bp_list[i]
            bp2 = bp_list[j]

            # Check if they are crossed
            if (bp1[0] < bp2[0] < bp1[1] < bp2[1]) or (bp2[0] < bp1[0] < bp2[1] < bp1[1]):
                crossed_pairs.append(bp1)
                crossed_pairs.append(bp2)
                crossed_pairs_set.add(bp1[0])
                crossed_pairs_set.add(bp1[1])
                crossed_pairs_set.add(bp2[0])
                crossed_pairs_set.add(bp2[1])
    return crossed_pairs, crossed_pairs_set

def dotbrackte2bp(structure):
    stack={'(':[],
           '[':[],
           '<':[],
           '{':[]}
    pop={')':'(',
         ']':'[',
         '>':"<",
         '}':'{'}       
    bp_list=[]
    matrix=np.zeros((len(structure),len(structure)))
    for i,s in enumerate(structure):
        if s in stack:
            stack[s].append((i,s))
        elif s in pop:
            forward_bracket=stack[pop[s]].pop()
            #bp_list.append(str(forward_bracket[0])+'-'+str(i))
            #bp_list.append([forward_bracket[0],i])
            bp_list.append([forward_bracket[0],i])

    return bp_list  


def calculate_f1_score_with_pseudoknots(true_pairs, predicted_pairs):
    true_pairs=[f"{i}-{j}" for i,j in true_pairs]
    predicted_pairs=[f"{i}-{j}" for i,j in predicted_pairs]
    
    true_pairs=set(true_pairs)
    predicted_pairs=set(predicted_pairs)

    # Calculate TP, FP, and FN
    TP = len(true_pairs.intersection(predicted_pairs))
    FP = len(predicted_pairs)-TP
    FN = len(true_pairs)-TP

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


# In[ ]:


from ast import literal_eval

F1s=[]
crossed_pair_F1s=[]
for true_bp, predicted_bp in zip(test_data['dbn'],hungarian_structures):
    predicted_bp=dotbrackte2bp(predicted_bp) 
    true_bp=dotbrackte2bp(true_bp) 
    crossed_pairs,crossed_pairs_set=detect_crossed_pairs(true_bp)
    predicted_crossed_pairs,predicted_crossed_pairs_set=detect_crossed_pairs(predicted_bp)
    
    _,_,f1=calculate_f1_score_with_pseudoknots(true_bp, predicted_bp)
    F1s.append(f1)
    

    if len(crossed_pairs)>0:
        _,_,crossed_pair_f1=calculate_f1_score_with_pseudoknots(crossed_pairs, predicted_crossed_pairs)
        crossed_pair_F1s.append(crossed_pair_f1)
    elif len(crossed_pairs)==0 and len(predicted_crossed_pairs)>0:
        crossed_pair_F1s.append(0)
    else:
        crossed_pair_F1s.append(np.nan)
    
    
print('global F1 mean',np.mean(F1s))
print('global F1 median',np.median(F1s))
print('crossed pair F1 mean',np.nanmean(crossed_pair_F1s))
print('crossed pair F1 median',np.nanmedian(crossed_pair_F1s))


# In[21]:


test_data['RibonanzaNet_Hungarian']=hungarian_structures
test_data['RibonanzaNet_Hungarian_F1']=F1s
test_data['RibonanzaNet_Hungarian_CP_F1']=crossed_pair_F1s

os.system('mkdir test_results')
test_data.to_parquet(f"bpRNA_finetuned_test.parquet",index=False)


# In[22]:

# test casp15
casp_data=pd.read_csv("../casp15.csv")
#casp_data.loc[2,'sequence']=casp_data.loc[2,'sequence'].replace('&','A')
casp_data['pairs']=[[] for _ in range(len(casp_data))]
casp_data['RNA_sequence']=casp_data['sequence']
casp_dataset=RNA2D_Dataset(casp_data)


# In[23]:


ribonanza_casp_ss=[]
model.eval()
for i in tqdm(range(len(casp_dataset))):
    example=casp_dataset[i]
    sequence=example['sequence'].cuda().unsqueeze(0)
    labels=example['ct'].cuda()

    with torch.no_grad():
        ribonanza_casp_ss.append(model(sequence).sigmoid().cpu().numpy())


# In[24]:


def dotbrackte2matrix(structure):
    stack={'(':[],
           '[':[],
           '<':[],
           '{':[]}
    pop={')':'(',
         ']':'[',
         '>':"<",
         '}':'{'}       
    bp_list=[]
    matrix=np.zeros((len(structure),len(structure)))
    for i,s in enumerate(structure):
        if s in stack:
            stack[s].append((i,s))
        elif s in pop:
            forward_bracket=stack[pop[s]].pop()
            bp_list.append(str(forward_bracket[0])+'-'+str(i))

            matrix[forward_bracket[0],i]=matrix[i,forward_bracket[0]]=1


    return matrix  


# In[25]:


casp_preds_hungarian=[]
casp_hungarian_structures=[]
casp_hungarian_bps=[]
for i in range(len(ribonanza_casp_ss)):
    s,bp=_hungarian(mask_diagonal(ribonanza_casp_ss[i][0]),theta=best_theta, min_len_helix=1)
    casp_hungarian_bps.append(bp)
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0],b[1]]=1
    ct_matrix=ct_matrix+ct_matrix.T
    casp_preds_hungarian.append(ct_matrix)
    casp_hungarian_structures.append(s)


# In[ ]:


from ast import literal_eval

F1s=[]
crossed_pair_F1s=[]
for true_bp, predicted_bp in zip(casp_data['bp'],casp_hungarian_bps):
    #predicted_bp=dotbrackte2bp(predicted_bp) 
    #true_bp=dotbrackte2bp(true_bp) 
    true_bp=literal_eval(true_bp)
    crossed_pairs,crossed_pairs_set=detect_crossed_pairs(true_bp)
    predicted_crossed_pairs,predicted_crossed_pairs_set=detect_crossed_pairs(predicted_bp)
    
    _,_,f1=calculate_f1_score_with_pseudoknots(true_bp, predicted_bp)
    F1s.append(f1)
    

    if len(crossed_pairs)>0:
        _,_,crossed_pair_f1=calculate_f1_score_with_pseudoknots(crossed_pairs, predicted_crossed_pairs)
        crossed_pair_F1s.append(crossed_pair_f1)
    elif len(crossed_pairs)==0 and len(predicted_crossed_pairs)>0:
        crossed_pair_F1s.append(0)
    else:
        crossed_pair_F1s.append(np.nan)
    
    
print('global F1 mean',np.mean(F1s))
print('global F1 median',np.median(F1s))
print('crossed pair F1 mean',np.nanmean(crossed_pair_F1s))
print('crossed pair F1 median',np.nanmedian(crossed_pair_F1s))


# In[26]:


casp_data['RibonanzaNet_Hungarian']=casp_hungarian_structures
casp_data['RibonanzaNet_Hungarian_F1']=F1s
casp_data['RibonanzaNet_Hungarian_CP_F1']=crossed_pair_F1s


# In[ ]:


casp_data.to_csv(f"bpRNA_casp15_ribonanzanet.csv",index=False)




# test casp16
casp16_data=pd.read_parquet("../../CASP16_SS_all_compiled.parquet")[['ID','structure']]
casp16_targets=pd.read_csv("../../3bead_model/casp16_sequences.csv").rename(columns={'sequence_id':'ID'})
casp16_targets=casp16_targets.merge(casp16_data,how='left',on='ID')
#only take those that have ID starts with R1 and the drop duplicates by sequence
casp16_targets=casp16_targets.loc[casp16_targets['ID'].str.startswith('R1')].drop_duplicates('sequence').reset_index(drop=True)
#drop ID==R1260
casp16_targets=casp16_targets.loc[casp16_targets['ID']!='R1260'].reset_index(drop=True)


casp16_targets['pairs']=[[] for _ in range(len(casp16_targets))]
casp16_targets['RNA_sequence']=casp16_targets['sequence']
casp_dataset=RNA2D_Dataset(casp16_targets)


ribonanza_casp_ss=[]
model.eval()
for i in tqdm(range(len(casp_dataset))):
    example=casp_dataset[i]
    sequence=example['sequence'].cuda().unsqueeze(0)
    labels=example['ct'].cuda()

    with torch.no_grad():
        ribonanza_casp_ss.append(model(sequence).sigmoid().cpu().numpy())

casp_preds_hungarian=[]
casp_hungarian_structures=[]
casp_hungarian_bps=[]
for i in range(len(ribonanza_casp_ss)):
    s,bp=_hungarian(mask_diagonal(ribonanza_casp_ss[i][0]),theta=best_theta, min_len_helix=1)
    casp_hungarian_bps.append(bp)
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0],b[1]]=1
    ct_matrix=ct_matrix+ct_matrix.T
    casp_preds_hungarian.append(ct_matrix)
    casp_hungarian_structures.append(s)

def find_dash_positions(s):
    return {pos for pos, char in enumerate(s) if char == '-'}

F1s=[]
crossed_pair_F1s=[]
for true_dbn, predicted_bp in zip(casp16_targets['structure'],casp_hungarian_bps):
    #predicted_bp=dotbrackte2bp(predicted_bp) 
    true_dbn=true_dbn[0]
    missing_positions=find_dash_positions(true_dbn)
    true_dbn=true_dbn.replace('-','.')
    true_bp=convert_dotbracket_to_bp_list(true_dbn,allow_pseudoknots=True)
    filtered_predicted_bp=[]
    for i in range(len(predicted_bp)):
        if predicted_bp[i][0] in missing_positions or predicted_bp[i][1] in missing_positions:
            pass
        else:
            filtered_predicted_bp.append(predicted_bp[i].copy())
    predicted_bp=filtered_predicted_bp
    #true_bp=literal_eval(true_bp)
    crossed_pairs,crossed_pairs_set=detect_crossed_pairs(true_bp)
    predicted_crossed_pairs,predicted_crossed_pairs_set=detect_crossed_pairs(predicted_bp)
    _,_,f1=calculate_f1_score_with_pseudoknots(true_bp, predicted_bp)
    F1s.append(f1)
    if len(crossed_pairs)>0:
        _,_,crossed_pair_f1=calculate_f1_score_with_pseudoknots(crossed_pairs, predicted_crossed_pairs)
        crossed_pair_F1s.append(crossed_pair_f1)
    elif len(crossed_pairs)==0 and len(predicted_crossed_pairs)>0:
        crossed_pair_F1s.append(0)
    else:
        crossed_pair_F1s.append(np.nan)

print('global F1 mean',np.mean(F1s))
print('global F1 median',np.median(F1s))
print('crossed pair F1 mean',np.nanmean(crossed_pair_F1s))
print('crossed pair F1 median',np.nanmedian(crossed_pair_F1s))

casp16_targets['RibonanzaNet_Hungarian']=casp_hungarian_structures
casp16_targets['RibonanzaNet_Hungarian_F1']=F1s
casp16_targets['RibonanzaNet_Hungarian_CP_F1']=crossed_pair_F1s


# In[ ]:


casp16_targets.to_csv(f"bpRNA_casp16_ribonanzanet.csv",index=False)