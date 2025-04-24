#for each folder of raw reads, corresponding to ABCDE, load all memmap and csv files for each prefix
#1. goin folder and glob
#2. get all prefixes associated with each experiment
#3. load all memmap and csv files for each prefix into a dictionary

import numpy as np
from glob import glob
import polars as pl
import os
from tqdm import tqdm
import h5py
#set POLARS_MAX_THREADS to 1
os.environ['POLARS_MAX_THREADS']="1"
from sklearn.model_selection import KFold, StratifiedKFold
from Functions import load_config_from_yaml

def load_rawread(hdf5_data,folder="../../Ribonanza2A_RawReads/tmp_merge_align/"):
    

    hdf5_data=h5py.File(hdf5_data, 'r')
    rawread_files=glob(folder+"*align_reads.txt.gz")

    prefixes=set([f.split('/')[-1].split('.')[0] for f in rawread_files if 'Nomod' not in f and 'nomod' not in f])
    print(prefixes)
    #prefixes=['RTB008_GenScript_DMS', 'RTB010_GenScript_2A3']
    seq_length=177
    memmap_data={}
    rawread_indices={}
    data={}
    
    #get reference and put in df
    # reference_sequences=[hdf5_data['sequences'][0,i].decode('utf-8') for i in range(hdf5_data['sequences'].shape[1])]
    # reference_sequences=pl.DataFrame(reference_sequences,columns=['sequence'])
    # print(reference_sequences[0])
    # exit()
    reference_sequences=list(hdf5_data['sequences'][0,:])
    reference_sequences= [s.decode('utf-8').replace('U','T') for s in reference_sequences]
    reference_sequences=pl.DataFrame({'sequence':reference_sequences})
    print(reference_sequences)
    for prefix in prefixes:
        print("loading rawread",prefix)

        #=pl.concat([pl.read_csv(f) for f in csv_files])
        memmap_files=glob(folder+prefix+"*.memmap")
        csv_files=glob(folder+prefix+"*.csv")
        csv_files.sort()
        memmap_files.sort()

        # if os.path.exists(prefix+"_meta.csv"):
        #     print("loading",prefix+"_meta.csv")
        #     meta_data=pl.read_csv(prefix+"_meta.csv")
        #     for csv,memmap in tqdm(zip(csv_files,memmap_files),total=len(csv_files)):
        #         memmap_data[memmap]=np.memmap(memmap,mode='r',dtype='uint8',shape=(csv['read_end'][-1],seq_length))
        # else:
        print("generating",prefix+"_meta.csv")

        
        meta_data=[]
        for csv,memmap in tqdm(zip(csv_files,memmap_files),total=len(csv_files)):
            #break
            assert csv.split('/')[-1].split('.')[0]==memmap.split('/')[-1].split('.')[0]

            csv=pl.read_csv(csv)
            csv=csv.with_columns(pl.lit(memmap).alias("memmap"))
            meta_data.append(csv)

            memmap_data[memmap]=np.memmap(memmap,mode='r',dtype='uint8',shape=(csv['read_end'][-1],seq_length))

        meta_data=pl.concat(meta_data)

        meta_data=reference_sequences.join(meta_data,on='sequence',how='left')

            #meta_data.write_csv(prefix+"_meta.csv")


        #check if there are nan values in the read_start and read_end columns
        print("checking for nan values in read_start and read_end")
        print(meta_data['read_start'].is_null().sum())
        print(meta_data['read_end'].is_null().sum())

        # exit()

        #start index is i-1, end index is j
        rawread_indices=[(memmap,i-1,j) for memmap,i,j in zip(meta_data['memmap'], meta_data['read_start'],meta_data['read_end'])]
        sequences=meta_data['sequence'].to_list()
        data[prefix]={'raw_data':memmap_data,'rawread_indices':rawread_indices,'sequences':sequences}
    # print(meta_data)
    # exit()
    #return memmap_data,rawread_indices
    #return {'raw_data':raw_data,'rawread_indices':rawread_indices,'sequences':sequences}
    
    return data, hdf5_data


def get_rawread_data(config):

    input_dir='../../input/'
    config = load_config_from_yaml("configs/pairwise.yaml")
    hdf_files=["Ribonanza2A_Genscript.v0.1.0.hdf5",
               'Ribonanza2B_full40B.v0.1.0.hdf5',
               'Ribonanza2C_full40B.v0.1.0.hdf5',
               'Ribonanza2D.v0.1.0.hdf5',
               'Ribonanza2E.v0.1.0.hdf5']

    raw_read_files=[["../../Ribonanza2A_RawReads/RTB008_GenScript_DMS.memmap","../../Ribonanza2A_RawReads/RTB010_GenScript_2A3.memmap"],
                      ["../../Ribonanza2B_RawReads/RTB004_Marathon_DMS.memmap","../../Ribonanza2B_RawReads/RTB006_SSII_2A3.memmap"],
                      ["../../Ribonanza2C_RawReads/RTB000_Marathon_DMS.memmap","../../Ribonanza2C_RawReads/RTB002_SSII_2A3.memmap"],
                      ["../../Ribonanza2D_RawReads/RTB000_Marathon_DMS.memmap","../../Ribonanza2D_RawReads/RTB002_SSII_2A3.memmap"],
                      ["../../Ribonanza2E_RawReads/RTB000_Marathon_DMS.memmap","../../Ribonanza2E_RawReads/RTB002_SSII_2A3.memmap"]] 

    snr_cutoff=1.0

    # rawread_data=[]
    # hdf5_data=[]
    all_data=[]
    all_train_indices=[]
    all_val_indices=[]
    cnt=0
    for hdf_file, memmap_files in zip(hdf_files,raw_read_files):
        print("Loading",hdf_file)
        #hdf5=pl.read_parquet(input_dir+hdf_file)
        #rawread,hdf5=load_rawread(input_dir+hdf_file,folder=folder)
        hdf5=h5py.File(input_dir+hdf_file, 'r')

        raw_read_data={}
        for f in memmap_files:
            print(f)
            metadata=pl.read_csv(f.replace(".memmap","_meta_data.csv"))
            #print(metadata.head())
            print(metadata['read_end_cat'].max()/1e9)
            #exit()

            nrows=metadata['read_end_cat'].max()

            memmap=np.memmap(f,mode='r',dtype='uint8',shape=(nrows,177))
            rawread_indices=[(i-1,j) for i,j in zip(metadata['read_start_cat'],metadata['read_end_cat'])]
            print(len(rawread_indices))
            raw_read_data[f]={'raw_data':memmap,'rawread_indices':rawread_indices}
        

        # print(raw_read_data.keys())
        # exit()
        # continue

        all_data.append([raw_read_data,hdf5])

        if hdf_file=="Ribonanza2A_Genscript.v0.1.0.hdf5": #keep part of A as val
            snr=hdf5['signal_to_noise'][:]
            high_quality_indices = np.where((snr>1.).sum(1)==2)[0]
            dirty_data_indices = np.where(((snr>snr_cutoff).sum(1)>=1)&((snr>1.).sum(1)!=2))[0]

            #dataset names
            sublib_data=pl.read_csv('../../sublib_id.csv')['sublibrary'].to_list()

            #StratifiedKFold on dataset
            kfold=StratifiedKFold(n_splits=config.nfolds,shuffle=True, random_state=0)
            fold_indices={}
            high_quality_dataname=[sublib_data[i] for i in high_quality_indices]
            for i, (train_index, test_index) in enumerate(kfold.split(high_quality_indices, high_quality_dataname)):
                fold_indices[i]=(high_quality_indices[train_index],high_quality_indices[test_index])
            #exit()

            train_indices=fold_indices[config.fold][0]
            val_indices=fold_indices[config.fold][1]




            train_indices=np.concatenate([train_indices,dirty_data_indices])

            print("train_indices",len(train_indices))
            print("val_indices",len(val_indices))
            #exit()
            #print(hdf_file)
            all_train_indices.extend([(cnt,i) for i in train_indices])
            all_val_indices.extend([(cnt,i) for i in val_indices])
        else:
            snr=hdf5['signal_to_noise'][:]
            dirty_data_indices = np.where(((snr>snr_cutoff).sum(1)>=1))[0]
            all_train_indices.extend([(cnt,i) for i in dirty_data_indices])
            print(f"{hdf_file} dirty data",len(dirty_data_indices))

        cnt+=1    


    print("train_indices",len(all_train_indices))
    print("val_indices",len(all_val_indices)) 

    return all_data, all_train_indices, all_val_indices


if __name__=="__main__":
    # data=load_rawread()
    # print(data.keys())  
    #print(rawread_indices.keys())
    input_dir='../../input/'
    config = load_config_from_yaml("configs/pairwise.yaml")
    hdf_files=["Ribonanza2A_Genscript.v0.1.0.hdf5",
               'Ribonanza2B_full40B.v0.1.0.hdf5',
               'Ribonanza2C_full40B.v0.1.0.hdf5',
               'Ribonanza2D.v0.1.0.hdf5',
               'Ribonanza2E.v0.1.0.hdf5']

    raw_read_files=[["../../Ribonanza2A_RawReads/RTB008_GenScript_DMS.memmap","../../Ribonanza2A_RawReads/RTB010_GenScript_2A3.memmap"],
                      ["../../Ribonanza2B_RawReads/RTB004_Marathon_DMS.memmap","../../Ribonanza2B_RawReads/RTB006_SSII_2A3.memmap"],
                      ["../../Ribonanza2C_RawReads/RTB000_Marathon_DMS.memmap","../../Ribonanza2C_RawReads/RTB002_SSII_2A3.memmap"],
                      ["../../Ribonanza2D_RawReads/RTB000_Marathon_DMS.memmap","../../Ribonanza2D_RawReads/RTB002_SSII_2A3.memmap"],
                      ["../../Ribonanza2E_RawReads/RTB000_Marathon_DMS.memmap","../../Ribonanza2E_RawReads/RTB002_SSII_2A3.memmap"]] 

    snr_cutoff=1.0

    # rawread_data=[]
    # hdf5_data=[]
    all_data=[]
    all_train_indices=[]
    all_val_indices=[]
    cnt=0
    for hdf_file, memmap_files in zip(hdf_files,raw_read_files):
        print("Loading",hdf_file)
        #hdf5=pl.read_parquet(input_dir+hdf_file)
        #rawread,hdf5=load_rawread(input_dir+hdf_file,folder=folder)
        hdf5=h5py.File(input_dir+hdf_file, 'r')

        raw_read_data={}
        for f in memmap_files:
            print(f)
            metadata=pl.read_csv(f.replace(".memmap","_meta_data.csv"))
            #print(metadata.head())
            print(metadata['read_end_cat'].max()/1e9)
            #exit()

            nrows=metadata['read_end_cat'].max()

            memmap=np.memmap(f,mode='r',dtype='uint8',shape=(nrows,177))
            rawread_indices=[(i-1,j) for i,j in zip(metadata['read_start_cat'],metadata['read_end_cat'])]
            
            raw_read_data[f]={'raw_data':memmap,'rawread_indices':rawread_indices}
        

        # print(raw_read_data.keys())
        # exit()
        # continue

        all_data.append([raw_read_data,hdf5])

        if hdf_file=="Ribonanza2A_Genscript.v0.1.0.hdf5": #keep part of A as val
            snr=hdf5['signal_to_noise'][:]
            high_quality_indices = np.where((snr>1.).sum(1)==2)[0]
            dirty_data_indices = np.where(((snr>snr_cutoff).sum(1)>=1)&((snr>1.).sum(1)!=2))[0]

            #dataset names
            sublib_data=pl.read_csv('../../sublib_id.csv')['sublibrary'].to_list()

            #StratifiedKFold on dataset
            kfold=StratifiedKFold(n_splits=config.nfolds,shuffle=True, random_state=0)
            fold_indices={}
            high_quality_dataname=[sublib_data[i] for i in high_quality_indices]
            for i, (train_index, test_index) in enumerate(kfold.split(high_quality_indices, high_quality_dataname)):
                fold_indices[i]=(high_quality_indices[train_index],high_quality_indices[test_index])
            #exit()

            train_indices=fold_indices[config.fold][0]
            val_indices=fold_indices[config.fold][1]




            train_indices=np.concatenate([train_indices,dirty_data_indices])

            print("train_indices",len(train_indices))
            print("val_indices",len(val_indices))
            #exit()
            #print(hdf_file)
            all_train_indices.extend([(cnt,i) for i in train_indices])
            all_val_indices.extend([(cnt,i) for i in val_indices])
        else:
            snr=hdf5['signal_to_noise'][:]
            dirty_data_indices = np.where(((snr>snr_cutoff).sum(1)>=1))[0]
            all_train_indices.extend([(cnt,i) for i in dirty_data_indices])
            print(f"{hdf_file} dirty data",len(dirty_data_indices))

        cnt+=1    


    print("train_indices",len(all_train_indices))
    print("val_indices",len(all_val_indices))