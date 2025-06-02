import yaml
import itertools
import os

# Hyperparameter grid to search
grid = {
    "epochs": [3, 5, 7],
    "cos_epoch": [-1],
    "loss_power_scale": [0.5, 1, 2],
    "upweight_positive": [1, 2],
    "batch_size": [1, 2],
    "optimizer_lr": [1e-4, 5e-5],
    "optimizer_weight_decay": [1e-4, 1e-5, 0.0],
}

# Output directories for configs, job scripts, and SLURM outputs
output_dir = "c1_grid_search_configs"
job_script_dir = "c1_grid_search_jobs"
slurm_out_dir = "c1_grid_search_slurm_out"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(job_script_dir, exist_ok=True)
os.makedirs(slurm_out_dir, exist_ok=True)

# Base configuration (single-level dictionary)
base_config = {
    "epochs": 5,
    "cos_epoch": 3,
    "loss_power_scale": 2,
    "upweight_positive": 2,
    "batch_size": 1,
    "gradient_clip": 10,
    "checkpoint_path": "finetuned_model_v8.pt",
    "normalize_length": 100,
    "device": "cuda",
    "model_name": "finetuned_RibonanzaNet",
    "model_pretrained": True,
    "model_config_path": "configs/pairwise.yaml",
    "optimizer_type": "Adam",
    "optimizer_lr": 0.0001,
    "optimizer_weight_decay": 0.0001,
    "scheduler_type": "CosineAnnealingLR",
    "cos_epoch": 3,
    "criterion_type": "BCEWithLogitsLoss",
    "criterion_reduction": "none",
    "validation_metric": "f1",
    "validation_save_best": True,
}

# SLURM job script template
job_script_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name={job_name}
#SBATCH --partition=defq
#SBATCH --output={slurm_out_dir}/{job_name}.out
#SBATCH --error={slurm_out_dir}/{job_name}.err

/lustre/fs0/scratch/shujun/miniconda3/envs/torch/bin/python finetune_C1_contact.py --config {config_path}
"""

# Generate grid search configs and job scripts
keys, values = zip(*grid.items())
for idx, combination in enumerate(itertools.product(*values)):
    # Copy the base config
    config = base_config.copy()

    # Update the configuration with grid search parameters
    for key, value in zip(keys, combination):
        config[key] = value

    # Save the updated configuration to a YAML file
    config_filename = os.path.join(output_dir, f"config_{idx+1}.yaml")
    with open(config_filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config: {config_filename}")

    # Generate a job script
    job_name = f"shujun-job-{idx+1}"
    job_script_content = job_script_template.format(
        job_name=job_name,
        config_path=config_filename,
        slurm_out_dir=slurm_out_dir
    )

    # Save the job script to a file
    job_script_filename = os.path.join(job_script_dir, f"job_{idx+1}.sh")
    with open(job_script_filename, "w") as f:
        f.write(job_script_content)
    print(f"Saved job script: {job_script_filename}")
