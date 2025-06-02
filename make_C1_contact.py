import pickle
import pandas as pd
import subprocess
import os

def run_cdhit_clustering(df, sequence_column, output_prefix, cdhit_executable, identity_threshold=0.8):
    """
    Perform CD-HIT clustering on sequences in a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing sequences in a column.
        sequence_column (str): Name of the column with sequences.
        output_prefix (str): Prefix for CD-HIT output files.
        cdhit_executable (str): Path to the CD-HIT executable.
        identity_threshold (float): Sequence identity threshold for clustering (default 0.8).

    Returns:
        pd.DataFrame: DataFrame with sequences and cluster information.
    """
    # Create a FASTA file from the sequence column
    input_fasta = f"{output_prefix}_input.fasta"
    output_fasta = f"{output_prefix}_output.fasta"
    cluster_file = f"{output_prefix}_output.fasta.clstr"

    with open(input_fasta, "w") as fasta_file:
        for idx, sequence in enumerate(df[sequence_column], start=1):
            fasta_file.write(f">seq{idx}\n{sequence}\n")

    # Run CD-HIT
    cdhit_command = [
        cdhit_executable,
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(identity_threshold),
        "-T", "0",  # Use all available threads
        "-M", "2000",  # Set memory limit to 2000 MB (greater than 1573 MB)
        "-l", "5"
    ]

    subprocess.run(cdhit_command, check=True)

    # Parse the clustering results
    clusters = {}
    current_cluster = None

    with open(cluster_file, "r") as clstr:
        for line in clstr:
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            #elif "*" in line:  # Representative sequence
            elif ">" in line:  # Sequence line
                seq_id = line.split(">")[1].split("...")[0]
                clusters[seq_id] = current_cluster

    # Map cluster information back to the DataFrame
    sequence_to_cluster = {f"seq{idx + 1}": clusters.get(f"seq{idx + 1}", None) for idx in range(len(df))}
    df["cluster"] = df.index.map(lambda idx: sequence_to_cluster.get(f"seq{idx + 1}"))

    # Clean up intermediate files
    # os.remove(input_fasta)
    # os.remove(output_fasta)
    # os.remove(cluster_file)

    return df

with open("train_data.pkl", "rb") as f:
    data = pickle.load(f)

#separate data into train, val, and annual test sets
# < 2018 is train, 2018-2020 is val, and onwards is test
data = pd.DataFrame({
    "sequence": data["sequence"],
    "temporal_cutoff": data["temporal_cutoff"],
    "description": data["description"],
    "all_sequences": data["all_sequences"],
    'xyz': data["xyz"]
})
data['pdb_id'] = data['description'].str.split('_').str[0]
data['year'] = data['temporal_cutoff'].str.split('-').str[0].astype(int)
data['year'] = data['year'].fillna(0).astype(int)
# Filter sequences based on year
train_sequences = data[data['year'] < 2018].reset_index(drop=True)
validation_sequences = data[data['year'].between(2018, 2019)].reset_index(drop=True)
print("Train sequences:", len(train_sequences))
print("Validation sequences:", len(validation_sequences))

test_sets=[]
for year in range(2019, 2025):
    test_set = data[data['year'] == year].reset_index(drop=True)
    test_sets.append(test_set)
    print(f"Test sequences for {year}: {len(test_set)}")

#run cdhit clustering on all sets
cdhit_executable = "../../cdhit/cd-hit"  # Adjust this path if necessary
train_sequences = run_cdhit_clustering(train_sequences, "sequence", "tmp", cdhit_executable)
validation_sequences = run_cdhit_clustering(validation_sequences, "sequence", "tmp_val", cdhit_executable)
for i, test_set in enumerate(test_sets):
    test_sets[i] = run_cdhit_clustering(test_set, "sequence", f"tmp_test_{2020 + i}", cdhit_executable)
# Save the clustered data
os.system("mkdir -p C1_contact")
train_sequences.to_pickle("C1_contact/train_sequences_clustered.pkl")
validation_sequences.to_pickle("C1_contact/validation_sequences_clustered.pkl")
for i, test_set in enumerate(test_sets):
    test_set.to_pickle(f"C1_contact/test_sequences_clustered_{2019 + i}.pkl")
