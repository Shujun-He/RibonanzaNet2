import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

def read_scores(score_dir):
    n = 216
    config_indices=range(1, n + 1)
    score_file_pattern=score_dir + "/c1_contact_grid_search_scores/config_{}.yaml.txt"

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
    sorted_indices = np.asarray(config_indices)[sorted_indices]

    scores_by_year={}
    for year in range(2019, 2025):
        scores_by_year[year] = []
        for index in sorted_indices:
            prefix = f"config_{index}.yaml"
            filename=f"{score_dir}/c1_contact_grid_search_results/{prefix}_test_f1_scores_{year}.pkl"

            data=pd.read_pickle(filename)

            cluster_f1s=[]
            for group in data.groupby('cluster'):
                group_data = group[1]
                cluster_f1s.append(group_data['f1_score'].mean())
            scores_by_year[year].append(np.mean(cluster_f1s))

    return scores_by_year


if __name__ == "__main__":
    base_dir = '/nrs/das/rnastruct/roi_exps'
    score_dirs = ['rnet2_plinit',
                  'rn2bcde_02M',
                  'rn2bcde_04M',
                  'rn2bcde_12M',
                  'rn2bcde_24M',
                  'rn2fgh_02M',
                  'rn2fgh_04M',
                  'rn2fgh_12M',
                  'rn2fgh_24M',
                  'rn2_random_02M',
                  'rn2_random_04M',
                  'rn2_random_12M',
                  'rn2_random_24M',
                  'rnet2alpha_baseline']

    colors = ['blue',
              'orange', 'orange', 'orange', 'orange',
              'green', 'green', 'green', 'green',
              'purple', 'purple', 'purple', 'purple',
              'brown']

    all_scores = []
    for score_dir in tqdm(score_dirs):
        scores_by_year = read_scores(base_dir + '/' + score_dir)
        all_scores.append(scores_by_year)

    fig, axes = plt.subplots(3, 2, figsize=(24, 16))

    for year, ax in zip(range(2019, 2025), axes.flatten()[:5]):
        data_to_plot = [scores_by_year[year] for scores_by_year in all_scores]
        box = ax.boxplot(data_to_plot, positions=np.arange(len(all_scores)) + 1)

        # Scatter plot (overlayed points)
        for ii in range(len(all_scores)):
            jitter = np.random.uniform(-0.1, 0.1, size=len(all_scores[ii][year]))
            ax.scatter(np.full_like(all_scores[ii][year], ii + 1) + jitter,
                        all_scores[ii][year], alpha=0.7, label=score_dirs[ii],
                        edgecolors='black', linewidths=0.5, color=colors[ii])
        ax.set_xticks(np.arange(1, len(all_scores) + 1))
        ax.set_xticklabels(score_dirs)
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylabel("Mean Cluster F1 Score")
        ax.set_title(f"{year} Cluster F1 Scores (Top 25% Configurations)")
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots/pl_exps_f1_scores_box_and_scatter_top25.png", dpi=300)
    plt.savefig("plots/pl_exps_f1_scores_box_and_scatter_top25.pdf")
    plt.close()

    print("Box + scatter plot saved to plots/pl_exps_f1_scores_box_and_scatter_top25.png")
