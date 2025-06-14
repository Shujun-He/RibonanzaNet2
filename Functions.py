import numpy as np
import csv
from os import path
import polars as pl
import yaml

import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import h5py



def load_and_split_rn2_ABCD():

    nfolds=6
    fold=0

    #first index is hdf file, and second is index in that hdf file
    hdf_files=[]
    hdf_train_indices=[]
    hdf_val_indices=[]


    #first get A and split into train val
    data=h5py.File('../../input/Ribonanza2A_Genscript.v0.1.0.hdf5', 'r')

    #get high snr data indices
    snr=data['signal_to_noise'][:]
    high_quality_indices = np.where((snr>1.).sum(1)==2)[0]
    dirty_data_indices = np.where(((snr>0.5).sum(1)>=1))[0]

    #dataset names
    sublib_data=pd.read_csv('../../sublib_id.csv')['sublibrary'].to_list()

    #StratifiedKFold on dataset
    kfold=StratifiedKFold(n_splits=nfolds,shuffle=True, random_state=0)
    fold_indices={}
    high_quality_dataname=[sublib_data[i] for i in high_quality_indices]
    for i, (train_index, test_index) in enumerate(kfold.split(high_quality_indices, high_quality_dataname)):
        fold_indices[i]=(high_quality_indices[train_index],high_quality_indices[test_index])
    #exit()

    train_indices=fold_indices[fold][0]
    val_indices=fold_indices[fold][1]

    train_indices=np.concatenate([train_indices,dirty_data_indices])

    print("train indices",len(train_indices))
    print("val indices",len(val_indices))

    hdf_files.append(data)
    hdf_train_indices.extend([(0,i) for i in train_indices])
    hdf_val_indices.extend([(0,i) for i in val_indices])

    #loop through BCD and use all for train
    BCD=['Ribonanza2B_full40B.v0.1.0.hdf5','Ribonanza2C_full40B.v0.1.0.hdf5','Ribonanza2D.v0.1.0.hdf5','Ribonanza2E.v0.1.0.hdf5']

    for file_index,hdf_file in zip(range(1,5),BCD):
        print("loading",
            hdf_file)
        print("file index",file_index)
        
        data=h5py.File('../../input/'+hdf_file, 'r')

        #get high snr data indices take any taht has one profile at snr>=1
        snr=data['signal_to_noise'][:]
        print(len(snr))
        train_indices = np.where((snr>0.5).sum(1)>=1)[0]
        
        print("train indices",len(train_indices))


        hdf_files.append(data)
        hdf_train_indices.extend([(file_index,i) for i in train_indices])


    print("total number of train indices",len(hdf_train_indices))
    print("total number of val indices",len(hdf_val_indices))

    return hdf_files,hdf_train_indices,hdf_val_indices

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

def load_state_dict_ignore_shape(model, pretrained_path):
    """
    Loads the state dictionary from the given path into the model,
    ignoring keys with mismatched weight shapes.
    
    Args:
        model (torch.nn.Module): The model to load the weights into.
        pretrained_path (str): The path to the saved state dictionary.
        
    Returns:
        None
    """
    # Get the current state dict from the model
    model_dict = model.state_dict()
    
    # Load the pretrained state dict
    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    
    # Filter the pretrained state dict to only include keys that exist in the model
    # and have matching shapes
    filtered_dict = {}
    for key, value in pretrained_dict.items():
        if key in model_dict:
            if model_dict[key].shape == value.shape:
                filtered_dict[key] = value
            else:
                print(f"Skipping key '{key}' due to shape mismatch: "
                      f"model shape {model_dict[key].shape} vs. pretrained shape {value.shape}")
        else:
            print(f"Skipping key '{key}' as it is not found in the current model.")
    
    # Update the model's state dict with the filtered dictionary and load it
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print("Model state dict loaded with matching parameters.")

def print_learning_rates(optimizer):
    """
    Prints the learning rate for each parameter group in the given optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to inspect.
    """
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group.get('lr', None)
        print(f"Learning rate for parameter group {i}: {lr}")


if __name__=='__main__':
    print(load_config_from_yaml("configs/pairwise.yaml"))
