import os

import numpy as np
import grain
import grain.sharding
import h5py


class RNADataSource(grain.sources.RandomAccessDataSource):
    def __init__(
        self,
        hdf_file_paths: list[str],
        indices,
        load_r_norm_err: bool,
    ):
        self.hdf_file_paths = hdf_file_paths
        self.hdf_files = [None] * len(hdf_file_paths)
        self.load_r_norm_err = load_r_norm_err

        self.indices = indices
        self.tokens = {nt: i for i, nt in enumerate("ACGU")}
        self.tokens["P"] = 4  # Padding token

    def _open_hdf5_file(self, file_index):
        if self.hdf_files[file_index] is None:
            self.hdf_files[file_index] = h5py.File(self.hdf_file_paths[file_index], "r")
        return self.hdf_files[file_index]

    def __repr__(self):
        # grain's Orbax checkpoint integration requires that the repr
        # match when restoring.
        return "RNADataSource"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        hdf_file_index, index = self.indices[idx]

        hdf_file = self._open_hdf5_file(hdf_file_index)

        sequence = hdf_file["sequences"][0, index].decode("utf-8")
        sequence = [self.tokens[nt] for nt in sequence]
        sequence = np.array(sequence, dtype=np.int32)

        seq_length = len(sequence)

        r_norm = hdf_file["r_norm"][index, :seq_length][:]

        if self.load_r_norm_err:
            r_norm_err = hdf_file["r_norm_err"][index, :seq_length][:]
        else:
            r_norm_err = None

        signal_to_noise = hdf_file["signal_to_noise"][index][:]

        data = {
            "sequence": sequence,
            "r_norm": r_norm,
            "r_norm_err": r_norm_err,
            "hdf_file_index": hdf_file_index,
            "signal_to_noise": signal_to_noise,
        }

        return data


def flip(data):
    """Flips an example, must be called after `AddLabels`."""
    data = data.copy()
    data["sequence"] = data["sequence"].flip(-1).copy(order="C")
    data["labels"] = data["labels"].flip(-2).copy(order="C")
    data["loss_masks"] = data["loss_masks"].flip(-2).copy(order="C")
    return data


class AddNoise(grain.transforms.RandomMap):
    """Adds random noise, must specify `load_r_norm_err=True`.

    Must be used before `AddLabels`."""

    def random_map(self, data, rng: np.random.Generator):
        data = data.copy()
        r_norm = data["r_norm"]
        r_norm_err = data.pop("r_norm_err")

        data["r_norm"] = r_norm + rng.normal(0, 1, r_norm_err.shape) * r_norm_err
        return data


class AddLabels(grain.transforms.Map):
    """Converts r_norm_err into the labels and loss_masks."""

    def __init__(self, num_hdf_files: int):
        self.num_hdf_files = num_hdf_files

    def map(self, data):
        num_hdf_files = self.num_hdf_files
        hdf_file_index = data["hdf_file_index"]
        signal_to_noise = data["signal_to_noise"]
        seq_length = len(data["sequence"])
        labels = np.full((seq_length, 2 * num_hdf_files), np.nan, dtype=np.float32)
        labels_experiment = data["r_norm"]
        labels[:, hdf_file_index * 2 : hdf_file_index * 2 + 2] = labels_experiment
        loss_mask = labels == labels  # mask nan labels

        label_mask = labels != labels

        labels[label_mask] = 0

        labels = labels.clip(0, 1)

        mask = np.ones(seq_length, dtype=np.int32)

        SN = np.zeros(self.num_hdf_files * 2, dtype=np.float32)
        SN[hdf_file_index * 2 : hdf_file_index * 2 + 2] = signal_to_noise

        return {
            "sequence": data["sequence"],
            "labels": labels,
            "masks": mask,
            "loss_masks": loss_mask,
            "SN": SN,
        }


class RandomFlip(grain.transforms.RandomMap):
    def random_map(self, data, rng: np.random.Generator):
        if rng.uniform() > 0.5:
            return flip(data)
        return data


class PadToMaxLen(grain.transforms.Map):
    def __init__(self, max_len: int):
        self.max_len = max_len

    def map(self, data):
        sequence = data["sequence"]
        labels = data["labels"]
        masks = data["masks"]
        loss_masks = data["loss_masks"]
        SN = data["SN"]

        seq_length = len(sequence)

        max_len = self.max_len
        assert max_len >= seq_length

        pad_len = max_len - seq_length
        pad_token_id = 4

        data = {
            "sequence": np.pad(sequence, (0, pad_len), constant_values=pad_token_id),
            "labels": np.pad(labels, ((0, pad_len), (0, 0)), constant_values=0),
            "masks": np.pad(masks, (0, pad_len), constant_values=0),
            "loss_masks": np.pad(loss_masks, ((0, pad_len), (0, 0)), constant_values=0),
            "SN": SN,
            "length": np.array(seq_length, dtype=np.int32),
        }
        return data


def make_data_loader(
    hdf_file_paths,
    indices,
    add_noise: bool,
    random_flip: bool,
    max_len: int,
    batch_size: int,
    shuffle: bool,
    shard_options: grain.sharding.ShardOptions,
    seed: int = 0,
    epochs: int | None = None,
):
    cpu_count = os.cpu_count()
    assert cpu_count is not None
    num_workers = min(batch_size // 2, cpu_count // 2, 8)

    if num_workers <= 0:
        num_workers = 1

    transformations = []
    if add_noise:
        transformations.append(AddNoise())
    transformations.append(AddLabels(num_hdf_files=len(hdf_file_paths)))
    if random_flip:
        transformations.append(RandomFlip())
    transformations.append(PadToMaxLen(max_len=max_len))
    return grain.load(
        source=RNADataSource(
            hdf_file_paths,
            indices,
            load_r_norm_err=add_noise,
        ),
        num_epochs=epochs,
        shuffle=shuffle,
        seed=0,
        transformations=transformations,
        batch_size=batch_size,
        worker_count=num_workers,
        shard_options=shard_options,
    )
