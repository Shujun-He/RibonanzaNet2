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


## Finetuning for C1' contact prediction


### data processing


C1_contact/  
├── train_sequences_clustered.pkl  
├── validation_sequences_clustered.pkl  
├── test_sequences_clustered_2019.pkl  
├── test_sequences_clustered_2020.pkl  
├── ...


#### 1. Requirements

- Python 3.7+
- `pandas`
- CD-HIT (compiled and executable from the command line)

#### 2. Input

First generate `train_data.pkl` with `preprocess_data.py` which requires Stanford3D folding data at https://www.kaggle.com/competitions/stanford-rna-3d-folding/data

The `make_C1_contact.py` script expects a `train_data.pkl` file containing a dictionary with:

- `sequence`: list of RNA/protein sequences
- `temporal_cutoff`: year strings (e.g., `"2017-01-01"`)
- `description`: PDB or sequence identifiers
- `all_sequences`: (optional, preserved)
- `xyz`: 3D coordinates (preserved for downstream use)

#### 3. Run the Script

Update the CD-HIT executable path in the script:

```python
cdhit_executable = "../../cdhit/cd-hit"  # Adjust path as needed
````

Then execute the script:

```bash
python make_C1_contact.py
```


### ⚙️ Processing Steps

1. **Temporal Split**
    
    - **Train**: Sequences from before 2018
        
    - **Validation**: Sequences from 2018–2019
        
    - **Test**: Yearly sets from 2019 to 2024
        
2. **Clustering with CD-HIT**
    
    - CD-HIT clusters sequences using an identity threshold of 0.8
        
    - Output cluster assignments are stored as a new `cluster` column in each DataFrame
        
3. **Saving Outputs**
    
    All processed subsets are saved under the `C1_contact/` directory as `.pkl` files for downstream model finetuning.
    


###  finetuning script


#### 🧪 C1-Contact Finetuning Script Description

This script performs finetuning of a pretrained `RibonanzaNet` model to predict **C1 contact maps** for RNA sequences. The workflow includes:

- **Loading clustered sequence data** (from CD-HIT clustering),
    
- **Filtering sequences by length**,
    
- **Constructing pairwise C1 contact maps** from atomic coordinates,
    
- **Training a pairwise prediction head** on top of a pretrained embedding model,
    
- **Hyperparameter tuning** using F1 score optimization over a validation set,
    
- **Evaluating model performance** on temporally-split yearly test sets from 2019–2024,
    
- **Saving best model weights and test results** per hyperparameter setting.
    

The script supports YAML-based configuration and optional debugging mode. It saves results and model checkpoints in the following directories:

- `c1_contact_grid_search_weights/`: best model weights per config
    
- `c1_contact_grid_search_scores/`: best validation F1 scores
    
- `c1_contact_grid_search_results/`: per-year test set F1 scores
    

Run this script using:

```bash
python finetune_c1_contact.py --config configs/your_config.yaml --max_len 1024
```

Enable debug mode for faster iteration:

```bash
python finetune_c1_contact.py --config configs/your_config.yaml --debug
```

---

### Plotting code
Use `compile_grid_search_c1.py` 




### example contact maps

use default cutoff of 15A  
see folder `c1_contact_examples/`  for examples  
looks ok but looks like adjacent nts in helices are also considered in contact  
maybe try adjusting cutoff  
alternatively it may be better to directly predict raw distances and compute contact F1s at different thresholds  

### To do

calculate F1 at top 3L predictions instead
mask |i-j| < 5 during F1 calculations (probabaly won't change F1 by much but is more correct)