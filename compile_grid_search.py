import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
import numpy as np

scores=[]
indices=[]
#for f in glob("grid_search_scores/*"):
n=216
for i in range(1,n+1):
    f=f"grid_search_scores/config_{i}.yaml.txt"

    with open(f,'r') as file:
        score=float(file.read())

    scores.append(score)

    #print(score)
    # if score>best_score:
    #     best_score=score
    #     best_config=f

scores=np.array(scores)

sorted_indices=np.argsort(scores)[::-1]

sorted_indices=sorted_indices[:int(len(sorted_indices)*0.25)]

#exit()

# Variables
#n = 216

# Lists for the CSV-based data
average_F1s_csv = []
average_CP_F1s_csv = []

# Lists for the Parquet-based data
average_F1s_parquet = []
average_CP_F1s_parquet = []

# Variables to track the best configuration
best_config_csv = None
best_score_csv = float('-inf')

best_config_parquet = None
best_score_parquet = float('-inf')

# Variables to track the best configuration
best_config_csv = None
best_score_csv = float('-inf')
best_avg_F1_csv = None
best_avg_CP_F1_csv = None

best_config_parquet = None
best_score_parquet = float('-inf')
best_avg_F1_parquet = None
best_avg_CP_F1_parquet = None

# ----- Process CSV Files -----
for i in tqdm(sorted_indices):
    try:
        file_name_csv = f"test_results/config_{i}.yaml_casp15_ribonanzanet.csv"
        df_csv = pd.read_csv(file_name_csv)
        
        avg_F1_csv = df_csv['RibonanzaNet_Hungarian_F1'].mean()
        avg_CP_F1_csv = df_csv['RibonanzaNet_Hungarian_CP_F1'].mean()
        total_score_csv = avg_F1_csv + avg_CP_F1_csv  # Calculate total score
        
        # Update best configuration for CSV
        if total_score_csv > best_score_csv:
            best_score_csv = total_score_csv
            best_config_csv = file_name_csv
            best_avg_F1_csv = avg_F1_csv
            best_avg_CP_F1_csv = avg_CP_F1_csv
        
        average_F1s_csv.append(avg_F1_csv)
        average_CP_F1s_csv.append(avg_CP_F1_csv)
    except Exception as e:
        print(f"Error processing CSV file {file_name_csv}: {e}")

# ----- Process Parquet Files -----
for i in tqdm(sorted_indices):
    try:
        file_name_parquet = f"test_results/config_{i}.yaml_finetuned_test.parquet"
        df_parquet = pd.read_parquet(file_name_parquet)
        
        avg_F1_parquet = df_parquet['RibonanzaNet_Hungarian_F1'].mean()
        avg_CP_F1_parquet = df_parquet['RibonanzaNet_Hungarian_CP_F1'].mean()
        total_score_parquet = avg_F1_parquet + avg_CP_F1_parquet  # Calculate total score
        
        # Update best configuration for Parquet
        if total_score_parquet > best_score_parquet:
            best_score_parquet = total_score_parquet
            best_config_parquet = file_name_parquet
            best_avg_F1_parquet = avg_F1_parquet
            best_avg_CP_F1_parquet = avg_CP_F1_parquet
        
        average_F1s_parquet.append(avg_F1_parquet)
        average_CP_F1s_parquet.append(avg_CP_F1_parquet)
    except Exception as e:
        print(f"Error processing Parquet file {file_name_parquet}: {e}")

# Print the best configurations
print(f"Best CSV configuration: {best_config_csv}")
print(f"  Total Score: {best_score_csv:.3f}")
print(f"  Average F1 Score: {best_avg_F1_csv:.3f}")
print(f"  Average CP F1 Score: {best_avg_CP_F1_csv:.3f}")

print(f"Best Parquet configuration: {best_config_parquet}")
print(f"  Total Score: {best_score_parquet:.3f}")
print(f"  Average F1 Score: {best_avg_F1_parquet:.3f}")
print(f"  Average CP F1 Score: {best_avg_CP_F1_parquet:.3f}")



# Calculate averages for CSV data
if len(average_F1s_csv) > 0:
    overall_avg_F1_csv = sum(average_F1s_csv) / len(average_F1s_csv)
    overall_avg_CP_F1_csv = sum(average_CP_F1s_csv) / len(average_CP_F1s_csv)
else:
    overall_avg_F1_csv = None
    overall_avg_CP_F1_csv = None

# Calculate averages for Parquet data
if len(average_F1s_parquet) > 0:
    overall_avg_F1_parquet = sum(average_F1s_parquet) / len(average_F1s_parquet)
    overall_avg_CP_F1_parquet = sum(average_CP_F1s_parquet) / len(average_CP_F1s_parquet)
else:
    overall_avg_F1_parquet = None
    overall_avg_CP_F1_parquet = None

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ----- Top row: CSV data -----
# Boxplot for CSV data (axes[0,0])
box_data_csv = [average_F1s_csv, average_CP_F1s_csv]
labels = ['Average F1s', 'Average CP F1s']
bp_csv = axes[0, 0].boxplot(box_data_csv, labels=labels, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', color='blue'),
                            medianprops=dict(color='red'),
                            whiskerprops=dict(color='blue'),
                            capprops=dict(color='blue'))
# Annotate CSV averages
if overall_avg_F1_csv is not None and overall_avg_CP_F1_csv is not None:
    axes[0, 0].text(1, overall_avg_F1_csv, f"Avg: {overall_avg_F1_csv:.3f}", ha='center', va='bottom', color='blue')
    axes[0, 0].text(2, overall_avg_CP_F1_csv, f"Avg: {overall_avg_CP_F1_csv:.3f}", ha='center', va='bottom', color='blue')

axes[0, 0].set_ylabel('Scores')
axes[0, 0].set_title('CASP15 Data: Boxplot of Average F1 and CP F1 Scores')

# Scatter plot for CSV data (axes[0,1])
axes[0, 1].scatter(average_F1s_csv, average_CP_F1s_csv, color='purple', alpha=0.7)
axes[0, 1].set_xlabel('Average F1 Scores')
axes[0, 1].set_ylabel('Average CP F1 Scores')
axes[0, 1].set_title('CASP15 Data: Scatter Plot of Average F1 vs CP F1')
axes[0, 1].set_xlim([0, 1])
axes[0, 1].set_ylim([0, 1])

# ----- Bottom row: Parquet data -----
# Boxplot for Parquet data (axes[1,0])
box_data_parquet = [average_F1s_parquet, average_CP_F1s_parquet]
bp_parquet = axes[1, 0].boxplot(box_data_parquet, labels=labels, patch_artist=True,
                                boxprops=dict(facecolor='lightgreen', color='green'),
                                medianprops=dict(color='red'),
                                whiskerprops=dict(color='green'),
                                capprops=dict(color='green'))
# Annotate Parquet averages
if overall_avg_F1_parquet is not None and overall_avg_CP_F1_parquet is not None:
    axes[1, 0].text(1, overall_avg_F1_parquet, f"Avg: {overall_avg_F1_parquet:.3f}", ha='center', va='bottom', color='green')
    axes[1, 0].text(2, overall_avg_CP_F1_parquet, f"Avg: {overall_avg_CP_F1_parquet:.3f}", ha='center', va='bottom', color='green')

axes[1, 0].set_ylabel('Scores')
axes[1, 0].set_title('PDB test Data: Boxplot of Average F1 and CP F1 Scores')

# Scatter plot for Parquet data (axes[1,1])
axes[1, 1].scatter(average_F1s_parquet, average_CP_F1s_parquet, color='orange', alpha=0.7)
axes[1, 1].set_xlabel('Average F1 Scores')
axes[1, 1].set_ylabel('Average CP F1 Scores')
axes[1, 1].set_title('PDB test Data: Scatter Plot of Average F1 vs CP F1')
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_ylim([0, 1])

# Adjust layout and save
plt.tight_layout()
plt.savefig("combined_2x2_plots.png")
plt.show()
