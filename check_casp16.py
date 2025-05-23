import pandas as pd

casp16_data = pd.read_parquet("../../CASP16_SS_all_compiled.parquet")[
    ["ID", "structure"]
]
casp16_targets = pd.read_csv("../../3bead_model/casp16_sequences.csv").rename(
    columns={"sequence_id": "ID"}
)
casp16_targets = casp16_targets.merge(casp16_data, how="left", on="ID")
# only take those that have ID starts with R1 and the drop duplicates by sequence
casp16_targets = (
    casp16_targets.loc[casp16_targets["ID"].str.startswith("R1")]
    .drop_duplicates("sequence")
    .reset_index(drop=True)
)
# drop ID==R1260
casp16_targets = casp16_targets.loc[casp16_targets["ID"] != "R1260"].reset_index(
    drop=True
)

for s, id in zip(casp16_targets["structure"], casp16_targets["ID"]):
    print("###")
    print(id)
    print(s)
