import argparse
from pathlib import Path
import os
from datetime import datetime


def bsub_directives(args: dict, idx: int) -> list[str]:
    """Generate LSF bsub directives based on provided arguments."""
    # Set job name based on node index
    jobname = args['job_name']
    if args['n_nodes'] > 1:
        if idx == 0:
            jobname += "-master"
        else:
            jobname += f"-worker{idx}"

    # Extract and validate gpu queue
    gpu_type = args["gpu_type"].lower()
    if gpu_type not in ['a100', 'h100', 'h200']:
        raise ValueError(f"Invalid gpu_type '{gpu_type}'. Must be one of: a100, h100, h200.")
    queue = f"gpu_{gpu_type}"
    max_gpus_per_node = 4 if gpu_type == 'a100' else 8
    if args['n_gpus_per_node'] > max_gpus_per_node:
        raise ValueError(
            f"n_gpus_per_node {args['n_gpus_per_node']} exceeds max "
            f"for {gpu_type} ({max_gpus_per_node})."
        )

    # Directives common to all jobs
    lines = [
        f"#BSUB -J {jobname}",
        "#BSUB -P das",
        f"#BSUB -q {queue}",
        f"#BSUB -n {args['n_gpus_per_node'] * args['n_cores_per_gpu']}",
        f"#BSUB -gpu \"num={args['n_gpus_per_node']}\"",
        f"#BSUB -oo run-logs/{jobname}.out",
        f"#BSUB -eo run-logs/{jobname}.err",
    ]

    # Request specific host for master node in multi-node runs
    if args['n_nodes'] > 1 and idx == 0:
        if not args['master_node']:
            raise ValueError(
                "master_node must be specified for multi-node runs. "
                f"Use `bhosts -w {args['gpu_type']}s` to find eligible nodes."
            )
        lines.append(f"#BSUB -m {args['master_node']}")

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
        lines.append("  --same_network \\")
        lines.append("  --use_fsdp \\")
        lines.append("  --fsdp_min_num_params 1000000 \\")
        lines.append("  --fsdp_auto_wrap_policy SIZE_BASED_WRAP \\")
        lines.append("  --fsdp_backward_prefetch NO_PREFETCH \\")
        lines.append("  --fsdp_sharding_strategy SHARD_GRAD_OP \\")
        lines.append("  --fsdp_state_dict_type FULL_STATE_DICT \\")
        lines.append("  --fsdp_use_orig_params true \\")


    # Additional arguments for multi-node runs
    if args['n_nodes'] > 1:
        lines.append('  --main_process_ip "$MASTER_ADDR" \\')
        lines.append('  --main_process_port "$PORT" \\')
        lines.append(f"  --machine_rank {idx} \\")

    return lines


def generate_all_launch_scripts(args: dict):
    """Generate all launch scripts for (possibly) multi-node training."""
    # Create output directories
    launch_dir = Path("lsf-scripts")
    launch_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("run-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    scripts = []

    # Generate configs for each node
    n_nodes = args['n_nodes']
    for idx in range(n_nodes):
        lines = ["#!/bin/bash", ""]

        lines.extend(bsub_directives(args, idx))

        lines.append("")
        lines.append("set -euo pipefail")
        lines.append("")
        if args['n_nodes'] > 1:
            lines.append("# Override defaults by setting environment variables before launching")
            lines.append("PORT=${PORT:-29500}")
            lines.append("")
        lines.append("export PYTHONUNBUFFERED=1")
        lines.append("export OMP_NUM_THREADS=8")
        lines.append("")

        if n_nodes > 1:
            lines.extend(nccl_env_vars())
            lines.append("")
            lines.append(
                f"MASTER_ADDR=$(getent ahostsv4 {args['master_node']} | awk 'NR==1{{print $1}}')"
            )
            if idx == 0:
                lines.append('echo "MASTER_ADDR=$MASTER_ADDR PORT=$PORT"')


        lines.append("")
        lines.append("accelerate launch \\")
        lines.extend(accelerate_args(args, idx))
        lines.append(f"  {args['script_name']} --config_path {args['config_path']}")

        # Write LSF script to file
        lsf_script = launch_dir / f"job_{idx:03d}.sh"
        with open(lsf_script, 'w', encoding="utf-8") as f:
            f.write('\n'.join(lines) + '\n')
        os.chmod(lsf_script, 0o755)
        scripts.append(lsf_script)

    print(f"\nGenerated all launch scripts in ./{launch_dir}")

    # Generate a master script to launch all jobs
    launch_all_script = Path("launch_all.sh")
    with open(launch_all_script, 'w', encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        for script in scripts:
            f.write(f"bsub < lsf-scripts/{script.name}\n")
    os.chmod(launch_all_script, 0o755)
    print("To launch all jobs, run: ./launch_all.sh")


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
        default=8,
        help="Number of GPUs per node (default: 8)"
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

    args = parser.parse_args()

    generate_all_launch_scripts(vars(args))


if __name__ == '__main__':
    main()
