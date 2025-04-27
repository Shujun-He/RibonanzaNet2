#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=shujun-job
#SBATCH --partition defq

# /lustre/fs0/scratch/shujun/miniconda3/envs/torch/bin/accelerate launch --config_file FSDP.yaml run.py

python finetune_3d.py