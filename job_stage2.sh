#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=shujun-job
#SBATCH --partition defq

#/lustre/fs0/scratch/shujun/miniconda3/envs/torch/bin/accelerate launch run.py

python eval_casp15.py
python eval_casp16.py