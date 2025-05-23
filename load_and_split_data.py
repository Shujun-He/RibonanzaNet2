import numpy as np
from sklearn.model_selection import KFold
import h5py


def load_and_split_rn2_ABCD(config):
    nfolds = config.nfolds
    fold = config.fold

    # first index is hdf file, and second is index in that hdf file
    hdf_files = []
    hdf_train_indices = []
    hdf_val_indices = []

    # first get A and split into train val
    data = h5py.File(config.hdf_files[0], "r")

    # get high snr data indices
    snr = data["signal_to_noise"][:]
    high_quality_indices = np.where((snr > 1.0).sum(1) == 2)[0]
    dirty_data_indices = np.where(((snr > 0.5).sum(1) >= 1))[0]

    # dataset names
    # sublib_data=pd.read_csv(config.subset_file)['sublibrary'].to_list()

    # StratifiedKFold on dataset
    kfold = KFold(n_splits=nfolds, shuffle=True, random_state=0)
    fold_indices = {}
    for i, (train_index, test_index) in enumerate(kfold.split(high_quality_indices)):
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

    # loop through rest of HDF5 files and use all for train

    for file_index in range(1, len(config.hdf_files)):
        hdf_file = config.hdf_files[file_index]
        print("loading", hdf_file)
        print("file index", file_index)

        data = h5py.File(hdf_file, "r")

        # get high snr data indices take any taht has one profile at snr>=1
        snr = data["signal_to_noise"][:]
        print(len(snr))
        train_indices = np.where((snr > 0.5).sum(1) >= 1)[0]

        print("train indices", len(train_indices))

        hdf_files.append(data)
        hdf_train_indices.extend([(file_index, i) for i in train_indices])

    print("total number of train indices", len(hdf_train_indices))
    print("total number of val indices", len(hdf_val_indices))

    return config.hdf_files, hdf_train_indices, hdf_val_indices


def dataset_dropout(dataset_name, train_indices, dataset2drop):
    # #dataset_name=pl.Series(dataset_name)
    # dataset_filter=pl.Series(dataset_name).str.starts_with(dataset2drop)
    # dataset_filter=dataset_filter.to_numpy()

    # dropout_indcies=set(np.where(dataset_filter==False)[0])
    # # print(dropout_indcies)
    # # exit()

    print(f"number of training examples before droppint out {dataset2drop}")
    print(train_indices.shape)
    before = len(train_indices)

    train_indices = [i for i in train_indices if dataset_name[i] != dataset2drop]
    train_indices = np.array(train_indices)

    print(f"number of training examples after droppint out {dataset2drop}")
    print(len(train_indices))
    after = len(train_indices)
    print(before - after, " sequences are dropped")

    # print(set([dataset_name[i] for i in train_indices]))
    # print(len(set([dataset_name[i] for i in train_indices])))
    # exit()

    return train_indices
