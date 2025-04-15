import re
import matplotlib.pyplot as plt

def extract_losses_from_slurm(slurm_file):
    pattern = re.compile(
        r"Epoch\s+\d+\s+Loss:\s+([0-9.]+)\s+Outer Product Loss:\s+([0-9.]+)\s+Raw Read Loss:\s+([0-9.]+),\s+Binary Loss:\s+([0-9.]+)\s+R Norm Loss:\s+([0-9.]+)"
    )

    losses = {
        "Loss": [],
        "Outer Product Loss": [],
        "Raw Read Loss": [],
        "Binary Loss": [],
        "R Norm Loss": [],
    }

    with open(slurm_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                losses["Loss"].append(float(match.group(1)))
                losses["Outer Product Loss"].append(float(match.group(2)))
                losses["Raw Read Loss"].append(float(match.group(3)))
                losses["Binary Loss"].append(float(match.group(4)))
                losses["R Norm Loss"].append(float(match.group(5)))

    return losses

def plot_losses(losses):
    for key, values in losses.items():
        plt.figure()
        plt.plot(values, label=key)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{key} over Time")
        plt.grid(True)
        plt.legend()
        key=key.replace(" ", "_").replace(":", "")
        plt.savefig(f"{key}_loss.png")

# Usage
losses = extract_losses_from_slurm("slurm-112456.out")
plot_losses(losses)
