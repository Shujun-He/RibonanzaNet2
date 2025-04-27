import pandas as pd
import matplotlib.pyplot as plt

# Simulated log data (since I don't have the actual file)
df = pd.read_csv("logs/fold0.csv")

# Create figure and subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot training loss
axes[0].plot(df["epoch"], df["train_loss"], marker='o', linestyle='-', color='b', label="Train Loss")
axes[0].set_ylabel("Train Loss")
axes[0].legend()
axes[0].grid(True)

# Plot validation loss
axes[1].plot(df["epoch"], df["val_loss"], marker='s', linestyle='-', color='r', label="Val Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val Loss")
axes[1].legend()
axes[1].grid(True)

# Show the plot
plt.tight_layout()
plt.savefig("loss.png")
