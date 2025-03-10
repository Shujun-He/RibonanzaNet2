import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "logs/fold0.csv"  # Update this path if necessary
df = pd.read_csv(file_path)#.iloc[6:]

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Train loss")

plt.subplot(122)
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='s')

# Formatting
plt.xlabel("Epoch")
plt.ylabel("Val loss")
plt.legend()
plt.grid()

# Show the plot
plt.savefig("loss.png")