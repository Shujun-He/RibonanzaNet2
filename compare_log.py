import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSV files
csv1_path = "logs/fold0.csv"  # Change this to the correct file path
csv2_path = "../test37/logs/fold0.csv"  # Change this to the correct file path

df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Ensure both files have the same columns
assert list(df1.columns) == list(df2.columns), "CSV files must have the same columns"

# Extract the epoch column
epochs = df1['epoch']
columns = df1.columns[1:]  # Exclude 'epoch'

# Create a 5x2 subplot layout
fig, axes = plt.subplots(5, 2, figsize=(12, 18))
fig.suptitle("Comparison of CSV Metrics Over Epochs", fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each column
for i, column in enumerate(columns):
    ax = axes[i]
    ax.plot(df1['epoch'], df1[column], label=f'CSV1 - {column}', marker='o')
    ax.plot(df2['epoch'], df2[column], label=f'CSV2 - {column}', marker='s')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(column)
    ax.set_title(column)
    ax.legend()
    ax.grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig("compare_log.png")