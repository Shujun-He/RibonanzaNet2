import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


def process_and_plot(
    config_indices,
    score_file_pattern,
    csv_pattern,
    parquet_pattern,
    additional_csv_pattern=None,
):
    scores = []

    # Read scores from files
    for i in config_indices:
        try:
            file_path = score_file_pattern.format(i)
            with open(file_path, "r") as file:
                scores.append(float(file.read()))
        except Exception as e:
            print(f"Error reading score file {file_path}: {e}")

    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1][: int(len(scores) * 0.25)]

    def process_files(pattern, sorted_indices):
        avg_F1s, avg_CP_F1s = [], []
        best_config, best_score, best_avg_F1, best_avg_CP_F1 = (
            None,
            float("-inf"),
            None,
            None,
        )

        for i in tqdm(sorted_indices):
            try:
                file_path = pattern.format(i)
                if pattern.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif pattern.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                else:
                    continue

                avg_F1 = df["RibonanzaNet_Hungarian_F1"].mean()
                avg_CP_F1 = df["RibonanzaNet_Hungarian_CP_F1"].mean()
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
    (
        avg_F1s_csv,
        avg_CP_F1s_csv,
        best_config_csv,
        best_score_csv,
        best_avg_F1_csv,
        best_avg_CP_F1_csv,
    ) = process_files(csv_pattern, sorted_indices)
    (
        avg_F1s_parquet,
        avg_CP_F1s_parquet,
        best_config_parquet,
        best_score_parquet,
        best_avg_F1_parquet,
        best_avg_CP_F1_parquet,
    ) = process_files(parquet_pattern, sorted_indices)

    if additional_csv_pattern:
        (
            avg_F1s_additional_csv,
            avg_CP_F1s_additional_csv,
            best_config_additional_csv,
            best_score_additional_csv,
            best_avg_F1_additional_csv,
            best_avg_CP_F1_additional_csv,
        ) = process_files(additional_csv_pattern, sorted_indices)
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

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    def plot_data(ax, data, overall_avg, labels, title, color):
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor=color, color=color),
            medianprops=dict(color="red"),
        )
        ax.set_title(title)
        ax.set_ylabel("Scores")
        for i, avg in enumerate(overall_avg):
            ax.text(
                i + 1, avg, f"Avg: {avg:.3f}", ha="center", va="bottom", color="blue"
            )

    def scatter_data(ax, x, y, title, xlabel, ylabel, color):
        ax.scatter(x, y, color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # Top row: CSV

    # Middle row: Parquet
    plot_data(
        axes[0, 0],
        [avg_F1s_parquet, avg_CP_F1s_parquet],
        [np.mean(avg_F1s_parquet), np.mean(avg_CP_F1s_parquet)],
        ["Average F1", "Average CP F1"],
        "PDB Data: Boxplot of Average F1 and CP F1 Scores",
        "lightgreen",
    )
    scatter_data(
        axes[0, 1],
        avg_F1s_parquet,
        avg_CP_F1s_parquet,
        "PDB Data: Scatter Plot of Average F1 vs CP F1",
        "Average F1",
        "Average CP F1",
        "orange",
    )

    plot_data(
        axes[1, 0],
        [avg_F1s_csv, avg_CP_F1s_csv],
        [np.mean(avg_F1s_csv), np.mean(avg_CP_F1s_csv)],
        ["Average F1", "Average CP F1"],
        "CASP15 Data: Boxplot of Average F1 and CP F1 Scores",
        "lightblue",
    )
    scatter_data(
        axes[1, 1],
        avg_F1s_csv,
        avg_CP_F1s_csv,
        "CASP15 Data: Scatter Plot of Average F1 vs CP F1",
        "Average F1",
        "Average CP F1",
        "purple",
    )

    # Bottom row: Additional CSV
    if additional_csv_pattern:
        plot_data(
            axes[2, 0],
            [avg_F1s_additional_csv, avg_CP_F1s_additional_csv],
            [np.mean(avg_F1s_additional_csv), np.mean(avg_CP_F1s_additional_csv)],
            ["Average F1", "Average CP F1"],
            "CASP16: Boxplot of Average F1 and CP F1 Scores",
            "lightcoral",
        )
        scatter_data(
            axes[2, 1],
            avg_F1s_additional_csv,
            avg_CP_F1s_additional_csv,
            "CASP16: Scatter Plot of Average F1 vs CP F1",
            "Average F1",
            "Average CP F1",
            "red",
        )

    plt.tight_layout()
    plt.savefig("combined_3x2_plots.png")
    plt.show()


# Example usage
n = 216
process_and_plot(
    config_indices=range(1, n + 1),
    score_file_pattern="grid_search_scores/config_{}.yaml.txt",
    csv_pattern="test_results/config_{}.yaml_casp15_ribonanzanet.csv",
    parquet_pattern="test_results/config_{}.yaml_finetuned_test.parquet",
    additional_csv_pattern="test_results/config_{}.yaml_casp16_ribonanzanet.csv",
)
