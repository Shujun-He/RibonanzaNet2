import numpy as np
import csv
from os import path
import polars as pl
import yaml

import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F

def rawread_training_loss(batch, model, accelerator):
    src=batch['sequence']#.cuda()
    masks=batch['mask'].bool()#.cuda()
    rawreads=batch['rawreads']#.cuda()
    rawreads_mask=batch['rawreads_mask']#.cuda()
    outer_product=batch['outer_products']#.cuda()
    r_norm=batch['r_norm']#.cuda()
    snr=batch['snr']#.cuda()  
    labels=rawreads[:,:,1:]

    src_expand=src.unsqueeze(1).repeat(1,rawreads.shape[1],1)
    change=(src_expand != rawreads[:,:,1:])

    l=change.shape[1]//2

    bs=len(src)
    #exit()
    loss=0
    snr_mask=snr>1.0
    with accelerator.autocast():
        output, binary_output, outer_product_pred, r_norm_pred=model(src,rawreads[:,:,:-1],masks) #BxNrxLrxC
        
        #get loss masks
        loss_masks =  (rawreads[:,:,1:] != 255) #mask out padding
        loss_masks = loss_masks.reshape(loss_masks.shape[0],2,-1,loss_masks.shape[-1])*snr_mask[:,:,None,None]
        loss_masks = loss_masks.reshape(loss_masks.shape[0],-1,loss_masks.shape[-1])
        
        #upweight mutated positions
        loss_weight = torch.ones_like(labels).float().to(output.device)
        loss_weight[(src_expand != rawreads[:,:,1:])]=100.0 #set weight to 10 for mismatched bases

        #compute raw read loss
        raw_read_loss=F.cross_entropy(output[loss_masks],labels[loss_masks],reduction='none')*loss_weight[loss_masks]
        raw_read_loss=raw_read_loss.mean()

        #total_raw_read_loss+=raw_read_loss.item()

        #binary loss
        binary_loss=F.binary_cross_entropy_with_logits(binary_output[loss_masks],change[loss_masks].float(),reduction='none')*loss_weight[loss_masks]
        binary_loss=binary_loss.mean()
        #loss+=binary_loss.mean()
        #total_binary_loss+=binary_loss.mean().item()

        #loss+=raw_read_loss

        outer_product_mask=torch.ones_like(outer_product).bool().permute(0,3,1,2)
        outer_product_mask=outer_product_mask.reshape(outer_product_mask.shape[0],2,-1,outer_product_mask.shape[-1],outer_product_mask.shape[-1])*snr_mask[:,:,None,None,None]
        outer_product_mask=outer_product_mask.reshape(outer_product_mask.shape[0],-1,outer_product_mask.shape[-2],outer_product_mask.shape[-1])
        outer_product_mask=outer_product_mask.permute(0,2,3,1)
        #~torch.isnan(outer_product)
        outer_product_loss=F.mse_loss(outer_product_pred[outer_product_mask],outer_product[outer_product_mask])
        # if outer_product_loss!=outer_product_loss:
        #     exit()
        #loss+=outer_product_loss*0.2

        #mae loss on r norm
        r_norm_mask=(~torch.isnan(r_norm))*snr_mask[:,None,:]
        r_norm_loss=F.l1_loss(r_norm_pred[r_norm_mask],r_norm[r_norm_mask])
        #loss+=r_norm_loss
        #total_r_norm_loss+=r_norm_loss.item()

        #total_outer_product_loss+=outer_product_loss.item()

    #return all losses for tracking
    return raw_read_loss, binary_loss, outer_product_loss, r_norm_loss

def plot_and_save_bar_chart(dataset_name, save_path):
    """
    Generates a bar chart for the counts of unique elements in dataset_name and saves the plot.

    Parameters:
        dataset_name (list): A list of elements to count and plot.
        save_path (str): Path to save the bar chart image.

    Returns:
        None
    """
    # Count the unique elements
    element_counts = Counter(dataset_name)
    
    # Extract elements and their counts
    elements = list(element_counts.keys())
    counts = list(element_counts.values())
    
    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(elements, counts)
    plt.xlabel('Elements')
    plt.ylabel('Count')
    plt.title('Count of Unique Elements in Dataset')
    
    plt.xticks(rotation=45, ha='right')

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()  # Close the plot to free memory
    print(f"Bar chart saved to {save_path}")


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of steps in one epoch (len(train_loader))
        final_lr: Target learning rate at the end of warmup
        
    Note:
        Despite the name, self.last_epoch inherited from _LRScheduler 
        actually counts steps, not epochs. It starts at -1 and is 
        incremented by 1 every time scheduler.step() is called.
    """
    def __init__(self, optimizer, total_steps, final_lr):
        self.total_steps = total_steps
        self.final_lr = final_lr
        super().__init__(optimizer)  # last_epoch=-1 by default

    def get_lr(self):
        # self.last_epoch is actually the current step number (starts at 0)
        current_step = self.last_epoch
        # Calculate current step's learning rate
        progress = float(current_step) / self.total_steps
        # Clip progress to avoid lr going above final_lr
        progress = min(1.0, progress)
        
        return [self.final_lr * progress for _ in self.base_lrs]


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def drop_pk5090_duplicates(df):
    pk50_filter=df['dataset_name'].str.starts_with('PK50')
    pk90_filter=df['dataset_name'].str.starts_with('PK90')
    no_pk_df=df.filter((~pk50_filter) & (~pk90_filter))
    pk50_df=df.filter(df['dataset_name'].str.starts_with('PK50_AltChemMap_NovaSeq'))
    pk90_df=df.filter(df['dataset_name'].str.starts_with('PK90_Twist_epPCR'))

    assert len(pk50_df)==2729*2
    assert len(pk90_df)==2173*2

    new_df=pl.concat([no_pk_df,pk50_df,pk90_df])

    return new_df

def dataset_dropout(dataset_name,train_indices, dataset2drop):

    # #dataset_name=pl.Series(dataset_name)
    # dataset_filter=pl.Series(dataset_name).str.starts_with(dataset2drop)
    # dataset_filter=dataset_filter.to_numpy()

    # dropout_indcies=set(np.where(dataset_filter==False)[0])
    # # print(dropout_indcies)
    # # exit()


    print(f"number of training examples before droppint out {dataset2drop}")
    print(train_indices.shape)
    before=len(train_indices)


    train_indices= [i for i in train_indices if dataset_name[i]!=dataset2drop]
    train_indices=np.array(train_indices)

    print(f"number of training examples after droppint out {dataset2drop}")
    print(len(train_indices))
    after=len(train_indices)
    print(before-after," sequences are dropped")


    # print(set([dataset_name[i] for i in train_indices]))
    # print(len(set([dataset_name[i] for i in train_indices])))
    # exit()

    return train_indices

def get_pl_train(pl_train, seq_length=457):

    print(f"before filtering pl_train has shape {pl_train.shape}")
    pl_train=pl_train.unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])
    print(f"after filtering pl_train has shape {pl_train.shape}")
    #seq_length=206

    label_names=["reactivity_{:04d}".format(number+1) for number in range(seq_length)]
    error_label_names=["reactivity_error_{:04d}".format(number+1) for number in range(seq_length)]

    sequences=pl_train.unique(subset=["sequence_id"],maintain_order=True)['sequence'].to_list()
    sequence_ids=pl_train.unique(subset=["sequence_id"],maintain_order=True)['sequence_id'].to_list()
    labels=pl_train[label_names].to_numpy().astype('float16').reshape(-1,2,seq_length).transpose(0,2,1)
    errors=np.zeros_like(labels).astype('float16')
    SN=pl_train['signal_to_noise'].to_numpy().astype('float16').reshape(-1,2)

    SN[:]=10 # set SN to 10 so they don't get masked

    data_dict = {
        'sequences': sequences,
        'sequence_ids': sequence_ids,
        'labels': labels,
        'errors': errors,
        'SN': SN,
    }

    return data_dict

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

def write_config_to_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def get_distance_mask(L):

    m=np.zeros((L,L))

    for i in range(L):
        for j in range(L):
            if abs(i-j)>0:
                m[i,j]=1/abs(i-j)**2
            elif i==j:
                m[i,j]=1
    return m

class CSVLogger:
    def __init__(self,columns,file):
        self.columns=columns
        self.file=file
        if not self.check_header():
            self._write_header()


    def check_header(self):
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header


    def _write_header(self):
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

    def log(self,row):
        if len(row)!=len(self.columns):
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file,"a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

if __name__=='__main__':
    print(load_config_from_yaml("configs/pairwise.yaml"))
