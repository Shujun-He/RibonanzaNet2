import h5py
from Dataset import *
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def load_and_split_rn2_ABCD():
    nfolds = 6
    fold = 0

    # first index is hdf file, and second is index in that hdf file
    hdf_files = []
    hdf_train_indices = []
    hdf_val_indices = []

    # first get A and split into train val
    data = h5py.File("../../input/Ribonanza2A_Genscript.v0.1.0.hdf5", "r")

    # get high snr data indices
    snr = data["signal_to_noise"][:]
    high_quality_indices = np.where((snr > 1.0).sum(1) == 2)[0]
    dirty_data_indices = np.where(((snr > 1.0).sum(1) == 1))[0]

    # dataset names
    sublib_data = pd.read_csv("../../sublib_id.csv")["sublibrary"].to_list()

    # StratifiedKFold on dataset
    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=0)
    fold_indices = {}
    high_quality_dataname = [sublib_data[i] for i in high_quality_indices]
    for i, (train_index, test_index) in enumerate(
        kfold.split(high_quality_indices, high_quality_dataname)
    ):
        fold_indices[i] = (
            high_quality_indices[train_index],
            high_quality_indices[test_index],
        )
    # exit()

    train_indices = fold_indices[fold][0]
    val_indices = fold_indices[fold][1]

    train_indices = np.concatenate([train_indices, dirty_data_indices])

    print("train indices", len(train_indices))
    print("val indices", len(val_indices))

    hdf_files.append(data)
    hdf_train_indices.extend([(0, i) for i in train_indices])
    hdf_val_indices.extend([(0, i) for i in val_indices])

    # loop through BCD and use all for train
    BCD = [
        "Ribonanza2B.v0.1.0.hdf5",
        "Ribonanza2C_first10B.v0.1.0.hdf5",
        "Ribonanza2D.v0.1.0.hdf5",
    ]

    for file_index, hdf_file in zip(range(1, 4), BCD):
        print("loading", hdf_file)
        print("file index", file_index)

        data = h5py.File("../../input/" + hdf_file, "r")

        # get high snr data indices take any taht has one profile at snr>=1
        snr = data["signal_to_noise"][:]
        print(len(snr))
        train_indices = np.where((snr > 1.0).sum(1) >= 1)[0]

        print("train indices", len(train_indices))

        hdf_files.append(data)
        hdf_train_indices.extend([(file_index, i) for i in train_indices])

    print("total number of train indices", len(hdf_train_indices))
    print("total number of val indices", len(hdf_val_indices))

    return hdf_files, hdf_train_indices, hdf_val_indices
