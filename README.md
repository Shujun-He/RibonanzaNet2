# RibonanzaNet

Training code for RibonanzaNet that uses Ribonanza1 onemill dataset. 



## Data Download

You just need an hdf5 file that is public available on kaggle and can be downloaded using the kaggle API: https://www.kaggle.com/datasets/shujun717/ribonanza1-onemill-hdf5

## Environment

Create the environment from the environment file ```env.yml```

```conda env create -f env.yml```

Install ranger optimizer

```conda activate torch```

```
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
cd Ranger-Deep-Learning-Optimizer
pip install -e .
``` 

## How to run
First activate environment ```conda activate torch```

generate scripts with ```generate_multinode_configs.sh```. You will need to set ```--n_nodes --n_gpus_per_node --master_node```. This script works for single node training as well. For single process training, ```python run.py --config_path configs/10M.yaml``` will work.


### Single node training
first setup accelerate config with ```accelerate config```
```accelerate launch run.py --config_path configs/pairwise.yaml```


### Multinode training

You can generate scripts with ```generate_multinode_configs.sh``` which has

```python generate_multinode_configs.py --n_nodes 32 --n_gpus_per_node 8 --master_node gpu002 --script_name run.py```

 and then ```launch_all.sh``` to launch distributed processes


## Outputs

The code will generate a log file in ```logs/fold0.csv``` and model weights will be saved to ```models``` folder. It will also generate a ```run_stats.json``` that records total runtime


## Configuration File

This section explains the various parameters and settings in the configuration file for RibonanzaNet

### Model Hyperparameters
- `learning_rate`: 0.001  
  The learning rate for the optimizer. Determines the step size at each iteration while moving toward a minimum of the loss function.

- `batch_size`: 2  
  Number of samples processed per GPU per batch. 

- `test_batch_size`: 8  
  Batch size used for testing the model per GPU per batch.

- `epochs`: 40  
  Total number of training epochs the model goes through.

- `dropout`: 0.05  
  The dropout rate for regularization to prevent overfitting. It represents the proportion of neurons that are randomly dropped out of the neural network during training.

- `weight_decay`: 0.0001  
  Regularization technique to prevent overfitting by penalizing large weights.

- `k`: 5
  1D Convolution kernel size

- `ninp`: 256  
  The size of the input dimension.

- `nlayers`: 9  
  Number of RibonanzaNet blocks.

- `nclass`: 2  
  Number of classes for classification tasks.

- `ntoken`: 5  
  Number of tokens (AUGC + padding/N token) used in the model.

- `nhead`: 8  
  The number of heads in multi-head attention models.

- `use_flip_aug`: true  
  Indicates whether flip augmentation is used during training/inference.

- `gradient_accumulation_steps`: 2  
  Number of steps to accumulate gradients before performing a backward/update pass.

- `use_triangular_attention`: false  
  Specifies whether to use triangular attention mechanisms in the model.

- `pairwise_dimension`: 64  
  Dimension of pairwise interactions in the model.

### Data Scaling
- `use_data_percentage`: 1  
  The fraction of data used from the dataset (1= full data training).

- `use_dirty_data`: true  
  Indicates whether to include training data that has only one of 2A3/DMS profiles with SN>1. 

### Other Configurations
- `fold`: 0  
  The current fold in use if the data is split into folds for cross-validation.

- `nfolds`: 6  
  Total number of folds for cross-validation.

- `input_dir`: "../../input/"  
  Directory for input data. Put ```train_data.csv```, ```test_sequences.csv```, and ```sample_submission.csv``` here. 

- `gpu_id`: "0"  
  Identifier for the GPU used for training. Useful in single-GPU setup.

- `hdf_files`: "/lustre/fs0/scratch/shujun/BothLanes_RawReads/OneMil.v0.1.0.hdf5"  
  path to the data


---

## File structure
 
```logs``` has the csv log file with train/val oss,
```models``` has model weights and optimizer states,
```oofs``` has the val predictions

## Finetuning for secondary structure

the script to use is `finetune_temporal_split.py` which has the following command line args  
 
`config`: finetuning hyperparameter yaml file    
`rnet_config`: RibonanzaNet pretraining config   
`train_test_data`: path to train test data (parquet file) (uploaded to `/finetuning_data/pdb_ss_data_w_pub_dates.parquet`)  
`casp15_data`: path to casp15 RNA SS data (uploaded to `/finetuning_data/casp15.csv`)  

To generate grid search configs (216 in total) use `generate_finetune_configs.py`.
Then if you have slurm `bash launch_finetune.sh` otherwise you have to manage these tasks given your hardware  
Next, to compile statistics from top 25% best hyperparamters (selected based on val), use `compile_grid_search_v2.py`




