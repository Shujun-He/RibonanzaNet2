def process_files(pattern, sorted_indices):
    avg_F1s, avg_CP_F1s = [], []
    best_config, best_score, best_avg_F1, best_avg_CP_F1 = None, float('-inf'), None, None

    for i in tqdm(sorted_indices):
        try:
            file_path = pattern.format(i)
            if pattern.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif pattern.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                continue

            avg_F1 = df['RibonanzaNet_Hungarian_F1'].mean()
            avg_CP_F1 = df['RibonanzaNet_Hungarian_CP_F1'].mean()
            total_score = avg_F1 + avg_CP_F1

            if total_score > best_score:
                best_score = total_score
                best_config = file_path
                best_avg_F1 = avg_F1
                best_avg_CP_F1 = avg_CP_F1

            avg_F1s.append(avg_F1)
            avg_CP_F1s.append(avg_CP_F1)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    #return avg_F1s, avg_CP_F1s#, best_config, best_score, best_avg_F1, best_avg_CP_F1
    return np.array(avg_F1s), np.array(avg_CP_F1s)

import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

folders=["../exps/test10","../exps/test42",'test14','test17']
labels=["Rnet1", "Ribonanza1 data",'Cross attn','Cross concat']
all_folder_metrics = {}
all_folder_cp_metrics = {}
#folder='test4'
n = 216

# Main processing loop
for folder in folders:
    scores = []
    # Read scores from files
    score_file_pattern = folder + "/grid_search_scores/config_{}.yaml.txt"
    for i in range(1, n + 1):
        try:
            file_path = score_file_pattern.format(i)
            with open(file_path, 'r') as file:
                scores.append(float(file.read()))
        except Exception as e:
            print(f"Error reading score file {file_path}: {e}")

    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1][:int(len(scores) * 0.25)]

    # Process metrics for each dataset
    casp15_F1s, casp15_CP_F1s = process_files(folder + "/test_results/config_{}.yaml_casp15_ribonanzanet.csv", sorted_indices)
    casp16_F1s, casp16_CP_F1s = process_files(folder + "/test_results/config_{}.yaml_casp16_ribonanzanet.csv", sorted_indices)
    pdb_F1s, pdb_CP_F1s = process_files(folder + "/test_results/config_{}.yaml_finetuned_test.parquet", sorted_indices)

    # Store average metrics for each folder
    all_folder_metrics[folder] = {
        "CASP15": casp15_F1s,
        "CASP16": casp16_F1s,
        "PDB": pdb_F1s
    }
    # Store CP_F1s for boxplot
    all_folder_cp_metrics[folder] = {
        "CASP15": casp15_CP_F1s,
        "CASP16": casp16_CP_F1s,
        "PDB": pdb_CP_F1s
    }


reference_F1_scores={"CASP16":0.79,"CASP15":0.936,"PDB":0.88}
reference_CP_F1_scores={"CASP16":0.46,"CASP15":0.7,"PDB":0.48}



# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 18), sharey=True)

categories = ["CASP15", "CASP16", "PDB"]

# Loop through each category for F1 and CP_F1 Scores
for i, category in enumerate(categories):
    # F1 Scores Boxplot
    f1_data = [all_folder_metrics[folder][category] for folder in folders]
    axes[i, 0].boxplot(f1_data, patch_artist=True, showmeans=True)
    axes[i, 0].set_title(f"F1 Scores: {category}", fontsize=12)
    axes[i, 0].set_xticks(range(1, len(folders) + 1))
    axes[i, 0].set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    if i == 1:
        axes[i, 0].set_ylabel("F1 Score", fontsize=12)
    # Add reference F1 score
    axes[i, 0].axhline(reference_F1_scores[category], color='red', linestyle='--', label="Rnet1 F1")
    axes[i, 0].legend(fontsize=8)

    # CP_F1 Scores Boxplot
    cp_f1_data = [all_folder_cp_metrics[folder][category] for folder in folders]
    axes[i, 1].boxplot(cp_f1_data, patch_artist=True, showmeans=True)
    axes[i, 1].set_title(f"CP_F1 Scores: {category}", fontsize=12)
    axes[i, 1].set_xticks(range(1, len(folders) + 1))
    axes[i, 1].set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    # Add reference CP_F1 score
    axes[i, 1].axhline(reference_CP_F1_scores[category], color='red', linestyle='--', label="Rnet1 CP_F1")
    axes[i, 1].legend(fontsize=8)

# Adjust layout and save the plot
fig.suptitle("Top 25% Grid Search F1 and CP_F1 Scores by Category", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('compare_crossconcat.png', dpi=300)
plt.show()