import pandas as pd
import matplotlib.pyplot as plt

# Sample log file data
start_epoch=10
df1 = pd.read_csv("log.csv").loc[start_epoch:]
df2 = pd.read_csv("../test10_rnet1/log.csv").loc[start_epoch:]
df3 = pd.read_csv("../test10_nopretrain/log.csv").loc[start_epoch:]


# Create subplots
metrics = ['train_loss', 'val_loss', 'val_rmsd', 'val_lddt']
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    axs[i].plot(df1['epoch'], df1[metric], label='Rnet2')
    axs[i].plot(df2['epoch'], df2[metric], label='Rnet1', linestyle='--')
    axs[i].plot(df3['epoch'], df3[metric], label='Rnet2_no_pretrained')
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel(metric)
    axs[i].set_title(f'{metric}')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig('comparison_rnet1.png')
