#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import yaml
import random

def generate_default_node_list(n_nodes: int) -> list:
    """Generate default node names in the format gpu001, gpu002, etc."""
    return [f"gpu{i+1:03d}" for i in range(n_nodes)]

def generate_accelerate_config(node_rank: int, n_nodes: int, n_gpus_per_node: int, 
                             main_node: str = None, base_port: int = 29500) -> dict:
    """Generate accelerate config for a specific node with FSDP support.
    
    Args:
        node_rank: Rank of the current node
        n_nodes: Total number of nodes
        n_gpus_per_node: Number of GPUs per node
        main_node: IP address of the main node (optional for FSDP)
        base_port: Base port number for communication
    
    Returns:
        dict: Accelerate configuration dictionary
    """
    config = {
        'compute_environment': 'LOCAL_MACHINE',
        'debug': False,
        'distributed_type': 'MULTI_GPU',
        'downcast_bf16': 'no',
        'enable_cpu_affinity': False,
        'machine_rank': node_rank,
        'main_process_ip': main_node,
        'main_process_port': base_port + 1,
        'main_training_function': 'main',
        'mixed_precision': 'bf16',
        'num_machines': n_nodes,
        'num_processes': n_nodes * n_gpus_per_node,
        'rdzv_backend': 'static',
        'same_network': True,
        'tpu_env': [],
        'tpu_use_cluster': False,
        'tpu_use_sudo': False,
        'use_cpu': False
    }
    
    return config

def generate_slurm_script(n_gpus: int, config_file: str,
                         python_env_path: str, script_name: str,
                         partition: str, job_name: str, master_node=None,) -> str:
    """
    Generate SLURM job script content.
    Only include nodelist parameter for gpu001.
    """
    # Base SLURM parameters
    slurm_params = [
        "#!/bin/bash",
        "#SBATCH --nodes 1",
        f"#SBATCH --gpus-per-node={n_gpus}",
        "#SBATCH --exclusive",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition {partition}"
    ]
    
    # Add nodelist only for gpu001
    if master_node is not None:
        slurm_params.append(f"#SBATCH --nodelist={master_node}")
    
    # Add the accelerate launch command
    command = f"\n\n{python_env_path}/bin/accelerate launch --config_file {config_file} {script_name}"
    
    return '\n'.join(slurm_params) + command

def generate_all_configs(n_nodes: int = None, node_list: list = None,
                        main_node: str = None,
                        n_gpus_per_node: int = None,
                        python_env_path: str = None,
                        base_output_dir: str = "distributed_config",
                        script_name: str = "run.py",
                        partition: str = "defq",
                        job_name: str = "distributed-job"):
    """Generate both accelerate configs and SLURM job scripts."""
    # if node_list is None:
    #     if n_nodes is None:
    #         raise ValueError("Either node_list or n_nodes must be provided")
    #     node_list = generate_default_node_list(n_nodes)
    # n_nodes = len(node_list)
    # main_node = node_list[0]  # Use first node as main node
    
    # Create output directories
    base_dir = Path(base_output_dir)
    config_dir = base_dir / "accelerate_configs"
    slurm_dir = base_dir / "slurm_scripts"
    config_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate configs for each node
    for idx in range(n_nodes):
        # Generate accelerate config
        config = generate_accelerate_config(
            node_rank=idx,
            n_nodes=n_nodes,
            n_gpus_per_node=n_gpus_per_node,
            main_node=main_node
        )
        config_path = config_dir / f"accelerate_config_node_{idx}.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Generate SLURM script
        if idx == 0:
            master_node = main_node
        else:
            master_node = None
        script_content = generate_slurm_script(
            master_node=master_node,
            n_gpus=n_gpus_per_node,
            config_file=f"accelerate_configs/accelerate_config_node_{idx}.yaml",
            python_env_path=python_env_path,
            script_name=script_name,
            partition=partition,
            job_name=f"{job_name}-{idx}"
        )
        script_path = slurm_dir / f"job_{idx}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        print(f"Generated config and job script for job {idx}")
    
    # Generate a launch script to start all jobs
    launch_script =  "launch_all.sh"
    with open(launch_script, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for idx in range(n_nodes):
            f.write(f"sbatch {slurm_dir}/job_{idx}.sh\n")
    os.chmod(launch_script, 0o755)
    
    print(f"\nGenerated all configs in {base_dir}")
    print(f"To launch all jobs, run: ./launch_all.sh")

def main():
    parser = argparse.ArgumentParser(description='Generate distributed training configs')
    parser.add_argument('--node_list', type=str, nargs='+',
                      help='List of node names (e.g., gpu001 gpu002). If not provided, will use gpu001-gpu{n_nodes}')
    parser.add_argument('--master_node', type=str, 
                      help='master_node')
    parser.add_argument('--n_nodes', type=int, default=None,
                      help='Number of nodes (used only if node_list is not provided)')
    parser.add_argument('--n_gpus_per_node', type=int, default=8,
                      help='Number of GPUs per node')
    parser.add_argument('--python_env_path', type=str, default='/lustre/fs0/scratch/shujun/miniconda3/envs/torch',
                      help='Path to Python environment (e.g., /lustre/fs0/scratch/user/miniconda3/envs/torch)')
    parser.add_argument('--config_dir', type=str, default='accelerate_configs',
                      help='Directory containing accelerate config files')
    parser.add_argument('--output_dir', type=str, default='.',
                      help='Directory to save job scripts')
    parser.add_argument('--script_name', type=str, default='run.py',
                      help='Name of the training script')
    parser.add_argument('--partition', type=str, default='defq',
                      help='SLURM partition to use')
    parser.add_argument('--job_name', type=str, default='distributed-job',
                      help='Base name for the job')
    
    args = parser.parse_args()
    
    generate_all_configs(
        n_nodes=args.n_nodes,
        node_list=args.node_list,
        main_node=args.master_node,
        n_gpus_per_node=args.n_gpus_per_node,
        python_env_path=args.python_env_path,
        base_output_dir=args.output_dir,
        script_name=args.script_name,
        partition=args.partition,
        job_name=args.job_name
    )

if __name__ == '__main__':
    main()