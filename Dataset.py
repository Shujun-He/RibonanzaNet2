from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
#import polars as pl
#tokens='ACGU().BEHIMSX'

def set_near_diagonal_to_nan(matrix, k=4):
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    n = matrix.shape[0]
    mask = np.abs(np.arange(n)[:, None] - np.arange(n)) < k
    matrix[mask] = np.nan
    return matrix

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

def compute_outer_product(rawread,sequence):
    sequence=sequence.unsqueeze(0)
    difference= (sequence!=rawread) * (rawread!=5)
    difference=F.one_hot(difference.long(),2).float()#[:,:,-1]
    nreads=len(difference)
    outer_product=torch.einsum('bic,bjd->ijcd',difference,difference)/nreads
    outer_product=outer_product.reshape(*outer_product.shape[:2],-1)
    return outer_product


class RawReadRNADataset(Dataset):
    def __init__(self,indices, data, max_len=192, max_seq=256):
        """
        raw read dataset
        tokens ACGUNP then start tokens for each experiment (DMS_nomod, DMS...etc)
        """
        self.indices=indices
        self.data=data
        #self.rnorm_data=rnorm_data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}
        self.tokens['P']=4
        self.tokens['T']=3
        #self.train=train
        self.max_len=max_len
        self.max_seq=max_seq


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #data[prefix]={'raw_data':raw_data,'csv':csv,'rawread_indices':rawread_indices,'sequences':sequences}

        batch_idx,idx=self.indices[idx]

        rawread_data=self.data[batch_idx][0]
        hdf5_data=self.data[batch_idx][1]

        all_rawreads=[]
        all_rawreads_mask=[]
        all_outer_products=[]
        all_mutation_frequency=[]
        for start_token,prefix in enumerate(rawread_data):
            #unpack data
            start_token=start_token+2*batch_idx

            rawread_indices=rawread_data[prefix]['rawread_indices']
            #sequences=rawread_data[prefix]['sequences']

            # print(idx)
            # print(len(rawread_indices))
            start,end=rawread_indices[idx]
            if end<start: #no raw reads
                rawreads=np.full((self.max_seq,self.max_len),255,dtype='int8')
            # else:
            #     rawreads=self.data[prefix]['raw_data'][file][start:end]
            elif end-start>self.max_seq:#randomly pick max_seq uniformly
                indices=np.random.choice(np.arange(start,end),self.max_seq,replace=False)
                rawreads=rawread_data[prefix]['raw_data'][indices]
            elif end-start<=self.max_seq:
                #padd with 4
                rawreads=rawread_data[prefix]['raw_data'][start:end]
                rawreads=np.pad(rawreads,((0,self.max_seq-(end-start)),(0,0)),mode='constant',constant_values=255)


            sequence=hdf5_data['sequences'][0,idx].decode("utf-8")

            sequence=[self.tokens[nt] for nt in sequence]
            sequence=np.array(sequence)

            seq_length=len(sequence)

            #expand sequence to max_Seq
            #expanded_sequence=np.stack([sequence for _ in range(self.max_seq)],0)

            #fill dots with expanded sequence
            #rawreads[rawreads==5]=expanded_sequence[rawreads==5]

            #if fewer than max_seq, pad with 4
            if seq_length<self.max_len:
                sequence=np.pad(sequence,(0,self.max_len-seq_length),mode='constant',constant_values=4)



            #rawreads=rawreads[:,:177]

            
            if (end-start)>1:
                outer_product=compute_outer_product(torch.tensor(rawread_data[prefix]['raw_data'][start:end]),torch.tensor(sequence))
                #outer_product=compute_outer_product(torch.tensor(rawreads),torch.tensor(sequence))
                outer_product=outer_product[:,:,3]-outer_product[:,:,1]*outer_product[:,:,2]
                outer_product=set_near_diagonal_to_nan(outer_product)
                outer_product=outer_product[:,:,None]
            else: #nans
                outer_product=torch.full((rawreads.shape[1],rawreads.shape[1],1),np.nan)

            #append start token 6 vector to the front of rawreads
            rawreads=np.concatenate([np.ones((rawreads.shape[0],1),dtype='int8')*(start_token+6),rawreads],axis=-1)
            

            N,L=rawreads.shape

            #pad to self.max_len with 4
            mask= sequence!=4
            rawreads_mask = rawreads!=255


            sequence=torch.tensor(sequence).long()
            rawreads=torch.tensor(rawreads).long()
            mask=torch.tensor(mask).float()
            rawreads_mask=torch.tensor(rawreads_mask).float()

            mutation_frequency=(rawreads[:,1:]!=sequence).float().mean(0)


            all_rawreads.append(rawreads)
            all_rawreads_mask.append(rawreads_mask)
            all_outer_products.append(outer_product)
            all_mutation_frequency.append(mutation_frequency)

        rawreads=torch.cat(all_rawreads)
        rawreads_mask=torch.cat(all_rawreads_mask)
        all_outer_products=torch.cat(all_outer_products,-1)
        
        #get r norm data
        #r_norm_index=self.rnorm_data['r_norm_index'][idx]
        r_norm=hdf5_data['r_norm'][idx]
        r_norm=torch.tensor(r_norm).float().clip(0,1)
        snr=hdf5_data['signal_to_noise'][idx]

        #put rnorm, snr, and outer product in multichannel format
        r_norm_mc=np.full((seq_length,2*len(self.data)),np.nan)
        r_norm_mc[:,batch_idx*2:batch_idx*2+2]=r_norm
        sn_mc=np.zeros(len(self.data)*2)
        sn_mc[batch_idx*2:batch_idx*2+2]=snr
        outer_product_mc=np.full((seq_length,seq_length,2*len(self.data)),np.nan)
        outer_product_mc[:,:,batch_idx*2:batch_idx*2+2]=all_outer_products

        r_norm_mc=torch.tensor(r_norm_mc).float().clip(0,1)
        sn_mc=torch.tensor(sn_mc).float()
        outer_product_mc=torch.tensor(outer_product_mc).float()


        data={'sequence':sequence,
              "rawreads":rawreads,
              "outer_products":all_outer_products,
              "mask":mask,
              "rawreads_mask":rawreads_mask,
              "r_norm":r_norm,
              "snr":snr,
              "snr_mc":sn_mc,
              "r_norm_mc":r_norm_mc,
              "outer_product_mc":outer_product_mc}


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


