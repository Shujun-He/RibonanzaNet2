import pandas as pd
from glob import glob
import os
import numpy as np

# Set the environment variable
os.environ["ARNIEFILE"] = "../arnie_file.txt"

# To check if the variable was set correctly, you can print it
print(os.environ["ARNIEFILE"])

from arnie.pk_predictors import _hungarian
from arnie.utils import convert_dotbracket_to_bp_list


def calculate_f1_score_with_pseudoknots(true_pairs, predicted_pairs):
    true_pairs = [f"{i}-{j}" for i, j in true_pairs]
    predicted_pairs = [f"{i}-{j}" for i, j in predicted_pairs]

    true_pairs = set(true_pairs)
    predicted_pairs = set(predicted_pairs)

    # Calculate TP, FP, and FN
    TP = len(true_pairs.intersection(predicted_pairs))
    FP = len(predicted_pairs) - TP
    FN = len(true_pairs) - TP

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1_score


def find_dash_positions(s):
    return {pos for pos, char in enumerate(s) if char == "-"}


def detect_crossed_pairs(bp_list):
    """
    Detect crossed base pairs in a list of base pairs in RNA secondary structure.

    Args:
    bp_list (list of tuples): List of base pairs, where each tuple (i, j) represents a base pair.

    Returns:
    list of tuples: List of crossed base pairs.
    """
    crossed_pairs_set = set()
    crossed_pairs = []
    # Iterate through each pair of base pairs
    for i in range(len(bp_list)):
        for j in range(i + 1, len(bp_list)):
            bp1 = bp_list[i]
            bp2 = bp_list[j]

            # Check if they are crossed
            if (bp1[0] < bp2[0] < bp1[1] < bp2[1]) or (
                bp2[0] < bp1[0] < bp2[1] < bp1[1]
            ):
                crossed_pairs.append(bp1)
                crossed_pairs.append(bp2)
                crossed_pairs_set.add(bp1[0])
                crossed_pairs_set.add(bp1[1])
                crossed_pairs_set.add(bp2[0])
                crossed_pairs_set.add(bp2[1])
    return crossed_pairs, crossed_pairs_set


dfs = glob("test_results/*_casp16_ribonanzanet*")

gt_ss = pd.read_parquet("../CASP16_GT_SS.parquet")

for df in dfs:
    df = pd.read_csv(df)
    merged = df[["ID", "RibonanzaNet_Hungarian"]].merge(
        gt_ss[["ID", "structure"]], on="ID", how="left"
    )

    F1s = []
    crossed_pair_F1s = []
    for i in range(len(merged)):
        predicted_dbn = merged.loc[i, "RibonanzaNet_Hungarian"]
        predicted_bp = convert_dotbracket_to_bp_list(
            predicted_dbn, allow_pseudoknots=True
        )
        true_dbn = merged.loc[i, "structure"]
        true_dbn = true_dbn[0]

        best_f1 = 0
        best_cp_f1 = 0
        for true_dbn in merged.loc[i, "structure"]:
            missing_positions = find_dash_positions(true_dbn)
            true_dbn = true_dbn.replace("-", ".")
            true_bp = convert_dotbracket_to_bp_list(true_dbn, allow_pseudoknots=True)
            filtered_predicted_bp = []
            for i in range(len(predicted_bp)):
                if (
                    predicted_bp[i][0] in missing_positions
                    or predicted_bp[i][1] in missing_positions
                ):
                    pass
                else:
                    filtered_predicted_bp.append(predicted_bp[i].copy())
            predicted_bp = filtered_predicted_bp
            # true_bp=literal_eval(true_bp)
            crossed_pairs, crossed_pairs_set = detect_crossed_pairs(true_bp)
            predicted_crossed_pairs, predicted_crossed_pairs_set = detect_crossed_pairs(
                predicted_bp
            )
            _, _, f1 = calculate_f1_score_with_pseudoknots(true_bp, predicted_bp)
            # F1s.append(f1)
            if len(crossed_pairs) > 0:
                _, _, crossed_pair_f1 = calculate_f1_score_with_pseudoknots(
                    crossed_pairs, predicted_crossed_pairs
                )

            elif len(crossed_pairs) == 0 and len(predicted_crossed_pairs) > 0:
                crossed_pair_f1 = 0
            else:
                crossed_pair_f1 = np.nan
            if f1 > best_f1:
                best_f1 = f1
                best_cp_f1 = crossed_pair_f1
        F1s.append(best_f1)
        crossed_pair_F1s.append(best_cp_f1)

    # print('global F1 mean',np.mean(F1s))
    # print('global F1 mean',np.mean(df['RibonanzaNet_Hungarian_F1']))
    print(np.mean(F1s) - np.mean(df["RibonanzaNet_Hungarian_F1"]))
    # print(np.nanmean(crossed_pair_F1s)-np.nanmean(df['RibonanzaNet_Hungarian_CP_F1']))
    # break
