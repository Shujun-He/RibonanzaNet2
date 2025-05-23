from glob import glob
import re
import pandas as pd
import numpy as np

results = []
for f in glob("grid_search_scores/*"):
    with open(f, "r") as file:
        score = float(file.read().strip())
    results.append((score, f))

# match = re.search(r'config_(\d+)\.yaml', filename)
# Sort results by score in descending order
results.sort(key=lambda x: x[0], reverse=True)

# Get the top 5 results
top5 = results[:3]

# Print the top 5 scores and their configuration file paths
for score, config in top5:
    print(f"Score: {score}, Config: {config}")

top5_number = [
    int(re.search(r"config_(\d+)\.yaml", config).group(1)) for score, config in top5
]
print(top5_number)

# for i in top5_number:
#     #test_results/config_126.yaml_finetuned_test.parquet


def print_best_of_5(template, id_col):
    # dfs=[pd.read_parquet(f'test_results/config_{i}.yaml_finetuned_test.parquet') for i in top5_number]
    if template.endswith("parquet"):
        dfs = [pd.read_parquet(template.format(i)) for i in top5_number]
    else:
        dfs = [pd.read_csv(template.format(i)) for i in top5_number]
    # dfs=[pd.read_parquet(template.format(i)) for i in top5_number]
    dfs = pd.concat(dfs)

    f1s = []
    cp_f1s = []

    for g in dfs.groupby(id_col):
        f1s.append(g[1]["RibonanzaNet_Hungarian_F1"].max())
        cp_f1s.append(g[1]["RibonanzaNet_Hungarian_CP_F1"].max())

    print(f"RibonanzaNet_Hungarian_F1: {sum(f1s) / len(f1s)}")
    print(f"RibonanzaNet_Hungarian_CP_F1: {np.nanmean(cp_f1s)}")


print("PDB test set")
template = "test_results/config_{}.yaml_finetuned_test.parquet"
print_best_of_5(template, "Sequence_ID")

print("CASP15")
template = "test_results/config_{}.yaml_casp15_ribonanzanet.csv"
print_best_of_5(template, "sequence_id")

print("CASP16")
template = "test_results/config_{}.yaml_casp16_ribonanzanet.csv"
print_best_of_5(template, "ID")
