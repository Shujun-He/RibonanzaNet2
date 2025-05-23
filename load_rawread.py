# for each folder of raw reads, corresponding to ABCDE, load all memmap and csv files for each prefix
# 1. goin folder and glob
# 2. get all prefixes associated with each experiment
# 3. load all memmap and csv files for each prefix into a dictionary

import numpy as np
from glob import glob
import polars as pl
import os
from tqdm import tqdm

# set POLARS_MAX_THREADS to 1
os.environ["POLARS_MAX_THREADS"] = "1"


def load_rawread(folder="../../Ribonanza2A_RawReads/tmp_merge_align/"):
    rawread_files = glob(folder + "*align_reads.txt.gz")

    prefixes = set([f.split("/")[-1].split(".")[0] for f in rawread_files])
    prefixes = ["RTB008_GenScript_DMS", "RTB010_GenScript_2A3"]
    seq_length = 177
    memmap_data = {}
    rawread_indices = {}
    data = {}
    for prefix in prefixes:
        print("loading rawread", prefix)
        memmap_files = glob(folder + prefix + "*.memmap")
        csv_files = glob(folder + prefix + "*.csv")
        csv_files.sort()
        memmap_files.sort()
        # =pl.concat([pl.read_csv(f) for f in csv_files])
        meta_data = []

        for csv, memmap in tqdm(zip(csv_files, memmap_files), total=len(csv_files)):
            # break
            assert (
                csv.split("/")[-1].split(".")[0] == memmap.split("/")[-1].split(".")[0]
            )

            csv = pl.read_csv(csv)
            csv = csv.with_columns(pl.lit(memmap).alias("memmap"))
            meta_data.append(csv)

            memmap_data[memmap] = np.memmap(
                memmap, mode="r", dtype="uint8", shape=(csv["read_end"][-1], seq_length)
            )

        meta_data = pl.concat(meta_data)

        # start index is i-1, end index is j
        rawread_indices = [
            (memmap, i - 1, j)
            for memmap, i, j in zip(
                meta_data["memmap"], meta_data["read_start"], meta_data["read_end"]
            )
        ]
        sequences = meta_data["sequence"].to_list()
        data[prefix] = {
            "raw_data": memmap_data,
            "rawread_indices": rawread_indices,
            "sequences": sequences,
        }

    # return memmap_data,rawread_indices
    # return {'raw_data':raw_data,'rawread_indices':rawread_indices,'sequences':sequences}

    return data


if __name__ == "__main__":
    data = load_rawread()
    print(data.keys())
    # print(rawread_indices.keys())
