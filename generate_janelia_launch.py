import argparse
from pathlib import Path
import os
from datetime import datetime
from dataclasses import dataclass


@dataclass
class GpuType:
    """Class representing GPU type and its properties."""
    name: str
    max_gpus_per_node: int
    max_cpus_per_node: int

    @property
    def queue(self) -> str:
        """Return the LSF queue name for this GPU type."""
        return f"gpu_{self.name}"

    @property
    def parallel_queue(self) -> str:
        """Return the LSF parallel queue name for this GPU type."""
        return self.queue + "_parallel"


# Available GPU types and their properties
GPU_TYPES = {
    'a100': GpuType(name='a100', max_gpus_per_node=4, max_cpus_per_node=48),
    'h100': GpuType(name='h100', max_gpus_per_node=8, max_cpus_per_node=96),
    'h200': GpuType(name='h200', max_gpus_per_node=8, max_cpus_per_node=96),
}


def bsub_directives(args: dict) -> list[str]:
    """Generate LSF bsub directives based on provided arguments."""
    # Set job name based on node index
    jobname = args['job_name']
    gpu_type = GPU_TYPES.get(args['gpu_type'].lower())

    # Directives common to all jobs
    n_cores_per_node = args['n_cores_per_gpu'] * args['n_gpus_per_node']
    n_total_cores = n_cores_per_node * args['n_nodes']
    lines = [
        f"#BSUB -J {jobname}",
        "#BSUB -P das",
        f"#BSUB -n {n_total_cores}",
        f'#BSUB -gpu "num={args['n_gpus_per_node']}"',
        f"#BSUB -oo run-logs/{jobname}.out",
        f"#BSUB -eo run-logs/{jobname}.err",
    ]

    # Additional directives depending on whether it's a multi-node job
    if args['n_nodes'] == 1:
        lines.append(f"#BSUB -q {gpu_type.queue}")
    else:
        lines.extend([
            f"#BSUB -q {gpu_type.parallel_queue}",
            f"#BSUB -R 'span[ptile={n_cores_per_node}]'",
        ])

    return lines


def nccl_env_vars() -> list[str]:
    """Generate NCCL environment variable exports for multi-node training."""
    lines = [
        "# NCCL settings optimized for ethernet",
        "export NCCL_DEBUG=INFO",
        "export NCCL_NSOCKS_PERTHREAD=4",
        "export NCCL_SOCKET_NTHREADS=4",
    ]

    return lines


def accelerate_args(args: dict, idx: int) -> list[str]:
    """Generate accelerate launch arguments based on provided settings."""
    # Base arguments
    lines = [
        f"  --num_processes {args['n_gpus_per_node'] * args['n_nodes']} \\",
        f"  --num_machines {args['n_nodes']} \\",
        f"  --mixed_precision {args['mixed_precision']} \\",
    ]

    # For more than 1 GPU, use FSDP (which is faster than DDP for our use case)
    if args['n_nodes'] * args['n_gpus_per_node'] > 1:
        lines.append("  --use_fsdp \\")
        lines.append("  --fsdp_min_num_params 1000000 \\")
        lines.append("  --fsdp_auto_wrap_policy SIZE_BASED_WRAP \\")
        lines.append("  --fsdp_backward_prefetch NO_PREFETCH \\")
        lines.append("  --fsdp_sharding_strategy SHARD_GRAD_OP \\")
        lines.append("  --fsdp_state_dict_type FULL_STATE_DICT \\")
        lines.append("  --fsdp_use_orig_params true \\")


    # Additional arguments for multi-node runs
    if args['n_nodes'] > 1:
        lines.append("  --same_network \\")
        lines.append('  --main_process_ip "$MASTER_ADDR" \\')
        lines.append('  --main_process_port "$PORT" \\')
        lines.append(f"  --machine_rank {idx} \\")

    return lines


def blaunch_command(args:dict, idx: int, postfix: str) -> list[str]:
    """Generate a blaunch command for multi-node execution."""
    log_name = f"run-logs/{args['job_name']}-{postfix}"

    lines = [f"# Launch command for {postfix}"]
    lines.append(f'blaunch -z ${{hosts[{idx}]}} "')
    lines.extend(nccl_env_vars())
    lines.append('${PYTHON_EXECUTABLE} accelerate launch \\')
    lines.extend(accelerate_args(args, idx))
    lines.append(f'  > {log_name}.out 2> {log_name}.err')
    lines.append('" &')

    return lines


def generate_all_launch_scripts(args: dict):
    """Generate all launch scripts for (possibly) multi-node training."""
    # Create output directories
    log_dir = Path("run-logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate configs for each node
    n_nodes = args['n_nodes']
    lines = ["#!/bin/bash", ""]

    # Add LSF directives
    lines.extend(bsub_directives(args))

    # Add some environment setup commands
    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("PYTHON_EXECUTABLE=$(which python)")
    lines.append("")
    lines.append("export PYTHONUNBUFFERED=1")
    lines.append("export OMP_NUM_THREADS=8")
    lines.append("")

    # For multi-node runs, set up master address and port
    if n_nodes > 1:
        lines.append("# Set up master address and port for multi-node training")
        lines.append('HOSTS=()')
        lines.append('for host in $(cat $LSB_DJOB_HOSTFILE | uniq); do')
        lines.append('    echo "Adding host: $host"')
        lines.append('    HOSTS+=($host)')
        lines.append('done')
        lines.append('echo Master node is ${hosts[0]}')
        lines.append("MASTER_ADDR=$(getent ahostsv4 ${hosts[0]} | awk 'NR==1{print $1}')")
        lines.append("")
        lines.append('CHECK="do while"')
        lines.append('while [[ ! -z $CHECK ]]; do')
        lines.append('    PORT=$(( ( RANDOM % 40000 )  + 20000 ))')
        lines.append('    CHECK=$(netstat -a | grep $PORT)')
        lines.append('done')
        lines.append('echo Master port is $PORT')
        lines.append("")

    # Add the accelerate launch command
    if n_nodes == 1:
        lines.append("# Single-node run, launch directly")
        lines.append("accelerate launch \\")
        lines.extend(accelerate_args(args, 0))
        lines.append(f"  {args['script_name']} --config_path {args['config_path']}")
    else:
        lines.append("# Multi-node run, launch via blaunch")
        lines.extend(blaunch_command(args, 0, "master"))
        n_leading_zeros = len(str(n_nodes - 1))
        for i in range(1, n_nodes):
            lines.append("")
            lines.extend(blaunch_command(args, i, f"worker{i:0{n_leading_zeros}d}"))

    # Write LSF script to file
    lsf_script = Path("launch.sh")
    with open(lsf_script, 'w', encoding="utf-8") as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(lsf_script, 0o755)

    print(f"Generated launch script 'launch.sh' for {n_nodes} node(s).")
    print("Use 'bsub < launch.sh' to submit the job.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate distributed training configs')
    parser.add_argument(
        "--master_node",
        type=str,
        help="host name for master node in multi-node runs "
        "(use `bmgroups` to find eligible nodes; ignored for single-node runs)",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--n_gpus_per_node",
        type=int,
        default=None,
        help="Number of GPUs per node (default: max for selected GPU type)"
    )
    parser.add_argument(
        "--n_cores_per_gpu",
        type=int,
        default=8,
        help="Number of CPU cores per GPU (default: 8)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file containing training parameters (required)",
    )
    parser.add_argument(
        "--script_name",
        type=str,
        default="run.py",
        help="Name of the training script (default: run.py)",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default='h200',
        help="Type of GPU to use (a100, h100, h200; default: h200)"
    )
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        "--job_name",
        type=str,
        default=f"rnet2-training-{datestr}",
        help="Base name for the job (default: rnet2-training-<timestamp>)"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        help="Mixed precision setting (fp16 or bf16; default: bf16)"
    )

    args = vars(parser.parse_args())

    # Extract and validate gpu queue
    gpu_type = GPU_TYPES.get(args['gpu_type'].lower())
    if args['n_gpus_per_node'] is None:
        args['n_gpus_per_node'] = gpu_type.max_gpus_per_node
    if gpu_type is None:
        raise ValueError(
            f"Invalid gpu_type '{args['gpu_type']}'. "
            f"Must be one of: {', '.join(GPU_TYPES.keys())}."
        )
    if args['n_gpus_per_node'] > gpu_type.max_gpus_per_node:
        raise ValueError(
            f"n_gpus_per_node {args['n_gpus_per_node']} exceeds max "
            f"for {gpu_type.name} ({gpu_type.max_gpus_per_node})."
        )
    if args['n_cores_per_gpu'] * args['n_gpus_per_node'] > gpu_type.max_cpus_per_node:
        raise ValueError(
            f"Total CPU cores per node ({args['n_cores_per_gpu'] * args['n_gpus_per_node']}) "
            f"exceeds max for {gpu_type.name} ({gpu_type.max_cpus_per_node})."
        )

    generate_all_launch_scripts(args)


if __name__ == '__main__':
    main()
