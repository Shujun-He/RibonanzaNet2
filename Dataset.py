from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
#import polars as pl
#tokens='ACGU().BEHIMSX'

def load_bpp(filename,seq_length=177):
    matrix = [[0.0 for x in range(seq_length)] for y in range(seq_length)]
 #   #matrix=0
    # data processing
  #  for line in open(filename):
   #     line = line.strip()
    #    if line == "":
     #       break
      #  i,j,prob = line.split()
       # matrix[int(j)-1][int(i)-1] = float(prob)
        #matrix[int(i)-1][int(j)-1] = float(prob)

    matrix=np.array(matrix)

    #ap=np.array(matrix).sum(0)
    return matrix

class RNADataset(Dataset):
    def __init__(self,indices,data_dict,k=5,train=True,flip=False):

        self.indices=indices
        self.data_dict=data_dict
        self.k=k
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        self.tokens['P']=4
        self.train=train
        self.flip=flip



    def generate_src_mask(self,L1,L2,k):
        mask=np.ones((k,L2),dtype='int8')
        for i in range(k):
            mask[i,L1+i+1-k:]=0
        return mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        idx=self.indices[idx]

        sequence=[self.tokens[nt] for nt in self.data_dict['sequences'][idx]]
        sequence=np.array(sequence)


        seq_length=len(sequence)

        #labels are in the order 2A3, DMS
        labels=self.data_dict['labels'][idx][:seq_length].copy()


        loss_mask = (labels==labels) #mask nan labels
        #assert len(loss_mask)==
       #loss_mask[seq_length:]=0 #mask padding tokens


        label_mask=labels!=labels

        labels[label_mask]=0

        labels=labels.clip(0,1)

        sequence=torch.tensor(sequence).long()
        labels=torch.tensor(labels).float()
        loss_mask=torch.tensor(loss_mask).bool()
        mask=torch.ones(seq_length)

        SN=torch.tensor(self.data_dict['SN'][idx]).float()




        if (self.train and np.random.uniform()>0.5) and self.flip:
            sequence=sequence.flip(-1)
            #attention_mask=attention_mask.flip(-1).flip(-2)
            #mask=mask.flip(-1)
            labels=labels.flip(-2)
            loss_mask=loss_mask.flip(-2)


        data={'sequence':sequence,
              "labels":labels,
              "mask":mask,
              "loss_mask":loss_mask,
              "SN":SN,}


        return data

class TestRNAdataset(RNADataset):
    def __getitem__(self, idx):
        
        idx=self.indices[idx]

        #id=self.ids[idx]

        #rows=self.df.loc[self.df['id']==id].reset_index(drop=True)
        #print()
        #idx=int(idx)
        #print(self.tokens)
        sequence=[self.tokens[nt] for nt in self.data_dict['sequences'][idx]]
        sequence=np.array(sequence)

        seq_length=len(sequence)
        sequence=torch.tensor(sequence).long()
        mask=torch.ones(seq_length)
        #errors=torch.tensor(errors).float()

        id=self.data_dict['sequence_ids'][idx]
        # bpp=load_bpp(f"../../bpp_files_v2.0.3/{id}.txt",len(sequence))
        # bpp=torch.tensor(bpp).float()
        data={'sequence':sequence,
              "mask":mask,}

        return data



class Custom_Collate_Obj:
    def __init__(self,max_len=None):
        self.max_len=max_len

    def __call__(self,data):
        # 
        length=[]
        for i in range(len(data)):
            length.append(len(data[i]['sequence']))
        if self.max_len>0:
            max_len=self.max_len
        else:
            max_len=max(length)
        #max_len=206

        sequence=[]
        labels=[]
        masks=[]
        loss_masks=[]
        errors=[]
        SN=[]
        use_bpp='bpp' in data[0]
        #print(use_bpp)
        #print(data['bpp'])
        if use_bpp:
            bpps=[]
        length=[]
        for i in range(len(data)):
            #to_pad=max_len-length[i]
            to_pad=max_len-len(data[i]['sequence'])
            length.append(len(data[i]['sequence']))
            #if to_pad>0:
            sequence.append(F.pad(data[i]['sequence'],(0,to_pad),value=4))
            #masks.append(data[i]['mask'])
            masks.append(F.pad(data[i]['mask'],(0,to_pad),value=0))
            loss_masks.append(F.pad(data[i]['loss_mask'],(0,0,0,to_pad),value=0))
            #print(data[i]['labels'].shape)
            labels.append(F.pad(data[i]['labels'],(0,0,0,to_pad),value=0))
            SN.append(data[i]['SN'])
            if use_bpp:
                bpps.append(F.pad(data[i]['bpp'],(0,to_pad,0,to_pad),value=0))


        sequence=torch.stack(sequence)
        labels=torch.stack(labels)#.permute(0,2,1)
        masks=torch.stack(masks)
        loss_masks=torch.stack(loss_masks)#.permute(0,2,1)
        SN=torch.stack(SN)
        if use_bpp:
            bpps=torch.stack(bpps)
        # print(sequence.shape)
        # print(labels.shape)
        # exit()

        length=torch.tensor(length)

        data={'sequence':sequence,
              "labels":labels,
              "masks":masks,
              "loss_masks":loss_masks,
              "SN":SN,
              "length":length}

        if use_bpp:
            data['bpps']=bpps

        return data

class Custom_Collate_Obj_test(Custom_Collate_Obj):

    def __call__(self,data):
        length=[]
        for i in range(len(data)):
            length.append(len(data[i]['sequence']))

        use_bpp='bpp' in data[0]
        if use_bpp:
            bpps=[]
        max_len=max(length)#+100
        #max_len=206
        sequence=[]
        masks=[]
        for i in range(len(data)):
            to_pad=max_len-length[i]
            sequence.append(F.pad(data[i]['sequence'],(0,to_pad),value=4))
            masks.append(F.pad(data[i]['mask'],(0,to_pad),value=0))
            #masks.append(F.pad(data[i]['mask'],(0,to_pad,0,0),value=0))
            if use_bpp:
                bpps.append(F.pad(data[i]['bpp'],(0,to_pad,0,to_pad),value=0))
        sequence=torch.stack(sequence)
        masks=torch.stack(masks)
        length=torch.tensor(length)
        

        data={'sequence':sequence,
              "masks":masks,
              "length":length,
              }
        
        if use_bpp:
            bpps=torch.stack(bpps)
            data["bpps"]=bpps

        return data
