# RibonanzaNet2 PL init pipeline

This branch uses Ribonanza1 PL data to initilize a RibonanzaNet2 model. 
Unlike other branches which use HDF5 files, this one uses an earlier version of the data pipeline with memmap files

The checkpoint is released here: https://www.kaggle.com/datasets/shujun717/rnet2-alpha-pl-init, but I created this branch so that PL init can be done for different model sizes 

## Data setup

### Training data

You need `train_data.csv` from https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data, which is real experimental data from Ribonanza1 that had >1 SNR  

`python process_data.py` to process the data



### PL Training data

You need https://www.kaggle.com/datasets/shujun717/merged-noisy-train-test-3parts, which are pseudo labeled data by top Kaggle models on sequences that had <1 SNR in experiments

`python make_pl_data.py` to process the data

To avoid padding to 457 (longest sequence in pl data), I only use sequences that are <= 207 to avoid excessive padding which shouldn't affect accuracy but makes training much faster.


## Training

Training script is `run_pl.py` which is multi-node ready, so it's compatible with other accelerate launch scripts. 
Training is done in 2 stages:  
1. PL data (1.8M) for 10 epochs
2. real data (210k) for 5 epochs


## Config

Config is `configs/pairwise.yaml`



