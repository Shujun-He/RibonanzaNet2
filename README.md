# RibonanzaNet

Training code for RibonanzaNet. 

Profile a small test example with `python run_profile.py --config_path configs/10M.yaml`.

# Example notebooks

You may not want to retrain RibonanzaNet from scratch and rather just use pretrained checkpoints, so we have created example notebooks: \
finetune: https://www.kaggle.com/code/shujun717/ribonanzanet-2d-structure-finetune \
secondary structure inference: https://www.kaggle.com/code/shujun717/ribonanzanet-2d-structure-inference \
chemical mapping inference: https://www.kaggle.com/code/shujun717/ribonanzanet-inference

## Data Download

You just need ```train_data.csv```, ```test_sequences.csv```, and ```sample_submission.csv``` from 
https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data

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

Set up accelerate with ```accelerate config``` in the terminal or with ```--config_path``` option

For an example of a accelerate config file, see ```accelerate_config.yaml```


### Data processing

run ```process_data.py```. for now the default data path is set to ```/lustre/fs0/scratch/shared/Ribonanza2A_Genscript.v0.1.0.hdf5```, which is the new data containing chemical mapping for 8 sequences sequences. Note that for training, we only use sequences that have SNR>1 which amounts to 3.6 million


### Training
```accelerate launch run.py --config_path configs/pairwise.yaml```


### Multinode training

You can generate scripts with ```generate_multinode_configs.sh``` which has

```python generate_multinode_configs.py --n_nodes 32 --n_gpus_per_node 8 --master_node gpu002 --script_name run.py```

 and then ```launch_all.sh``` to launch distributed processes


### Inference
```accelerate launch inference.py --config_path configs/pairwise.yaml```

### Process raw prediction into submission file for Ribonanza
```python make_submission.py --config_path configs/pairwise.yaml```


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

---

## File structure
 
```logs``` has the csv log file with train/val oss,
```models``` has model weights and optimizer states,
```oofs``` has the val predictions



