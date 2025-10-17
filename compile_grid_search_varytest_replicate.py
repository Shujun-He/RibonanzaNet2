import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


def process(config_indices, score_file_pattern, csv_pattern, parquet_pattern, additional_csv_pattern=None):
    scores = []

    # Read scores from files
    for i in config_indices:
        try:
            file_path = score_file_pattern.format(i)
            with open(file_path, 'r') as file:
                scores.append(float(file.read()))
        except Exception as e:
            print(f"Error reading score file {file_path}: {e}")

    # scores = np.array(scores)
    # sorted_indices = np.argsort(scores)[::-1]#[:int(len(scores) * 0.25)]
    sorted_indices = config_indices

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

        return avg_F1s, avg_CP_F1s, best_config, best_score, best_avg_F1, best_avg_CP_F1

    # Process CSV and Parquet files
    avg_F1s_csv, avg_CP_F1s_csv, best_config_csv, best_score_csv, best_avg_F1_csv, best_avg_CP_F1_csv = process_files(csv_pattern, sorted_indices)
    avg_F1s_parquet, avg_CP_F1s_parquet, best_config_parquet, best_score_parquet, best_avg_F1_parquet, best_avg_CP_F1_parquet = process_files(parquet_pattern, sorted_indices)

    if additional_csv_pattern:
        avg_F1s_additional_csv, avg_CP_F1s_additional_csv, best_config_additional_csv, best_score_additional_csv, best_avg_F1_additional_csv, best_avg_CP_F1_additional_csv = process_files(additional_csv_pattern, sorted_indices)
        print(f"Best Additional CSV configuration: {best_config_additional_csv}")
        print(f"  Total Score: {best_score_additional_csv:.3f}")
        print(f"  Average F1 Score: {best_avg_F1_additional_csv:.3f}")
        print(f"  Average CP F1 Score: {best_avg_CP_F1_additional_csv:.3f}")

    # Print best configurations
    print(f"Best CSV configuration: {best_config_csv}")
    print(f"  Total Score: {best_score_csv:.3f}")
    print(f"  Average F1 Score: {best_avg_F1_csv:.3f}")
    print(f"  Average CP F1 Score: {best_avg_CP_F1_csv:.3f}")

    print(f"Best Parquet configuration: {best_config_parquet}")
    print(f"  Total Score: {best_score_parquet:.3f}")
    print(f"  Average F1 Score: {best_avg_F1_parquet:.3f}")
    print(f"  Average CP F1 Score: {best_avg_CP_F1_parquet:.3f}")

    return [avg_F1s_csv, avg_CP_F1s_csv, avg_F1s_parquet, avg_CP_F1s_parquet]


def plot_data(ax, data, overall_avg, labels, title, color, ylim):
    # bp = ax.boxplot(data, labels=labels, patch_artist=True,
    #                 boxprops=dict(facecolor=color, color=color),
    #                 medianprops=dict(color='red'))
    vp = ax.violinplot(data, showmeans=True, showmedians=True,
                       quantiles=[[0.25, 0.75] for _ in data], showextrema=False)
    for ii,pc in enumerate(vp['bodies']):
        pc.set_facecolor(color[ii])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    vp['cmedians'].set_color('red')
    vp['cquantiles'].set_color('red')
    # vp['cmaxes'].set_color('red')
    ax.set_title(title)
    ax.set_ylabel('Scores')
    ax.set_ylim(ylim)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([label.split('/')[-1] for label in labels])
    for i, avg in enumerate(overall_avg):
        ax.text(i + 1, avg, f"Avg: {avg:.3f}", ha='center', va='bottom', color='blue')
    ax.tick_params(axis='x', rotation=30)


if __name__ == "__main__":
    exp_dirs = ['/nrs/das/rnastruct/roi_exps/rnet2_plinit',
                '/nrs/das/rnastruct/roi_exps/rnet2alpha_baseline',
                '/groups/flyem/home/huangg/proj/rnaxjanelia/RibonanzaNet2-ft/rnet2alpha_rep',
                '/groups/flyem/home/huangg/proj/rnaxjanelia/RibonanzaNet2-ft/rnet2alpha_rep_bs6',
                '/groups/flyem/home/huangg/proj/rnaxjanelia/RibonanzaNet2-ft/rnet2alpha_rep_bs6_opt']

    plot_labels = ['pl_init', 'rnet2alpha', 'bs=3','bs=6','bs=6 run2']
    best_configs = [51, 63, 207, 210, 171]

    all_results = []
    for ii, exp_dir in enumerate(exp_dirs):
        results = process(range(60, 361, 5),
                          f"{exp_dir}/gs_scores/config_{best_configs[ii]}.yaml_{{:03d}}.txt",
                          f"{exp_dir}/test2_results/config_{best_configs[ii]}.yaml_{{:03d}}_casp15_ribonanzanet.csv",
                          f"{exp_dir}/test2_results/config_{best_configs[ii]}.yaml_{{:03d}}_finetuned_test.parquet")
        all_results.append(results)

    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    avg_F1s_parquet = [results[2] for results in all_results]
    avg_CP_F1s_parquet = [results[3] for results in all_results]
    avg_F1s_csv = [results[0] for results in all_results]
    avg_CP_F1s_csv = [results[1] for results in all_results]

    colors = ['lightgreen',
              'lightgreen',
              'lightblue',
              'lightblue',
              'lightblue',
              'lightblue']

    plot_data(axes[0, 0],
              avg_F1s_parquet,
              [np.mean(ff) for ff in avg_F1s_parquet],
              plot_labels,
              'PDB Data: Average F1 Scores', colors, (0.8, 1))
    plot_data(axes[0, 1],
              avg_CP_F1s_parquet,
              [np.mean(ff) for ff in avg_CP_F1s_parquet],
              plot_labels,
              'PDB Data: Average CP F1 Scores', colors, (0, 1))
    plot_data(axes[1, 0],
              avg_F1s_csv,
              [np.mean(ff) for ff in avg_F1s_csv],
              plot_labels,
              'CASP15 Data: Average F1 Scores', colors, (0.8, 1))
    plot_data(axes[1, 1],
              avg_CP_F1s_csv,
              [np.mean(ff) for ff in avg_CP_F1s_csv],
              plot_labels,
              'CASP15 Data: Average CP F1 Scores', colors, (0, 1))

    plt.tight_layout()
    plt.savefig("rep_varytest_combined_crossexps_plots.pdf")
    plt.savefig("rep_varytest_combined_crossexps_plots.png")
