from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import polars as pl
#tokens='ACGU().BEHIMSX'



class RNADataset(Dataset):
    def __init__(self,hdf_files,indices,train=True,flip=False,add_noise=False):

        self.hdf_files=hdf_files
        self.indices=indices
        #self.k=k
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        self.tokens['P']=4
        self.train=train
        self.flip=flip
        self.add_noise=add_noise


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        hdf_file_index, index =self.indices[idx]
        hdf_file=self.hdf_files[hdf_file_index]

        sequence=hdf_file['sequences'][0,index].decode("utf-8")

        sequence=[self.tokens[nt] for nt in sequence]
        sequence=np.array(sequence)
        # print(sequence)
        # exit()

        seq_length=len(sequence)

        #labels are in the order DMS, 2A3
        #init labels to nan with Lx(2*len(self.hdf_files))
        labels=np.full((seq_length,2*len(self.hdf_files)),np.nan)
        labels_experiment=hdf_file['r_norm'][index,:seq_length][:]

        
        if self.train and self.add_noise:
            #plt.plot(labels_experiment[:,0].clip(0,1),label='real')
            error=hdf_file['r_norm_err'][index,:seq_length][:]
            labels_experiment=labels_experiment+np.random.normal(0,1,error.shape)*error
            #plt.plot(labels_experiment[:,0].clip(0,1),label='noised')
        # plt.legend()
        # plt.savefig(f'noised/{idx}.png')
        # plt.close()
        #exit()

        labels[:,hdf_file_index*2:hdf_file_index*2+2]=labels_experiment

        # print(labels)
        # print(labels.shape)


        loss_mask = (labels==labels) #mask nan labels



        label_mask=labels!=labels

        labels[label_mask]=0

        labels=labels.clip(0,1)

        sequence=torch.tensor(sequence).long()
        labels=torch.tensor(labels).float()
        loss_mask=torch.tensor(loss_mask).bool()
        mask=torch.ones(seq_length)

        SN=np.zeros(len(self.hdf_files)*2)
        SN[hdf_file_index*2:hdf_file_index*2+2]=hdf_file['signal_to_noise'][index][:]
        SN=torch.tensor(SN).float()

        # print(SN.shape)
        # print(SN)
        # exit()



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
