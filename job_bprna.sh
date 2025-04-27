#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=shujun-job
#SBATCH --partition defq

/lustre/fs0/scratch/shujun/miniconda3/envs/torch/bin/accelerate launch finetune_bprna.py --config grid_search_configs/config_213.yaml
