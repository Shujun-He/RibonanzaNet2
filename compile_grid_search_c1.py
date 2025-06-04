import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

n = 216
config_indices=range(1, n + 1)
score_file_pattern="c1_contact_grid_search_scores/config_{}.yaml.txt"




# Read scores from files
scores=[]
for i in config_indices:
    try:
        file_path = score_file_pattern.format(i)
        with open(file_path, 'r') as file:
            scores.append(float(file.read()))
    except Exception as e:
        print(f"Error reading score file {file_path}: {e}")

scores = np.array(scores)
sorted_indices = np.argsort(scores)[::-1][:int(len(scores) * 0.25)]

# print(scores)
# exit()

#f"c1_contact_grid_search_results/{prefix}_test_f1_scores_{year}.pkl"

scores_by_year={}
for year in range(2019, 2025):
    scores_by_year[year] = []
    for index in sorted_indices:
        prefix = f"config_{index}.yaml"
        filename=f"c1_contact_grid_search_results/{prefix}_test_f1_scores_{year}.pkl"

        data=pd.read_pickle(filename)

        cluster_f1s=[]
        for group in data.groupby('cluster'):
            group_data = group[1]
            cluster_f1s.append(group_data['f1_score'].mean())
        scores_by_year[year].append(np.mean(cluster_f1s))
        #print(np.mean(cluster_f1s))
        #exit()


# Plot
plt.figure(figsize=(12, 6))

years = sorted(scores_by_year.keys())
data_to_plot = [scores_by_year[year] for year in years]

# Boxplot
box = plt.boxplot(data_to_plot, positions=np.arange(len(years)) + 1)

# Scatter plot (overlayed points)
for i, year in enumerate(years):
    jitter = np.random.uniform(-0.1, 0.1, size=len(scores_by_year[year]))
    plt.scatter(np.full_like(scores_by_year[year], i + 1) + jitter,
                scores_by_year[year], alpha=0.7, label=str(year), edgecolors='black', linewidths=0.5)

plt.xticks(ticks=np.arange(1, len(years) + 1), labels=years)
plt.xlabel("Year")
plt.ylabel("Mean Cluster F1 Score")
plt.title("Cluster F1 Scores by Year (Top 25% Configurations)")
plt.grid(True, linestyle='--', alpha=0.5)

# Save plot
os.makedirs("plots", exist_ok=True)
plt.tight_layout()
plt.savefig("plots/f1_scores_box_and_scatter_top25.png", dpi=300)
plt.close()

print("Box + scatter plot saved to plots/f1_scores_box_and_scatter_top25.png")