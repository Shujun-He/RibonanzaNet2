import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm

# set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter": 10,
    "structural_violation_epoch": 50,
    "balance_weight": False,
}


# Load data

train_sequences = pd.read_csv("../3d_data/train_sequences.csv")
train_labels = pd.read_csv("../3d_data/train_labels.csv")

train_labels["pdb_id"] = train_labels["ID"].apply(
    lambda x: x.split("_")[0] + "_" + x.split("_")[1]
)
train_labels["pdb_id"]

all_xyz = []

for pdb_id in tqdm(train_sequences["target_id"]):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    # break
    xyz = df[["x_1", "y_1", "z_1"]].to_numpy().astype("float32")
    xyz[xyz < -1e17] = float("Nan")
    all_xyz.append(xyz)


# filter the data
# Filter and process data
filter_nan = []
max_len = 0
for xyz in all_xyz:
    if len(xyz) > max_len:
        max_len = len(xyz)

    # fill -1e18 masked sequences to nans

    # sugar_xyz = np.stack([nt_xyz['sugar_ring'] for nt_xyz in xyz], axis=0)
    filter_nan.append(
        (np.isnan(xyz).mean() <= 0.5)
        & (len(xyz) < config["max_len_filter"])
        & (len(xyz) > config["min_len_filter"])
    )

print(f"Longest sequence in train: {max_len}")

filter_nan = np.array(filter_nan)
non_nan_indices = np.arange(len(filter_nan))[filter_nan]

train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
all_xyz = [all_xyz[i] for i in non_nan_indices]

# pack data into a dictionary

data = {
    "sequence": train_sequences["sequence"].to_list(),
    "temporal_cutoff": train_sequences["temporal_cutoff"].to_list(),
    "description": train_sequences["description"].to_list(),
    "all_sequences": train_sequences["all_sequences"].to_list(),
    "xyz": all_xyz,
}

# Split data into train and test
all_index = np.arange(len(data["sequence"]))
cutoff_date = pd.Timestamp(config["cutoff_date"])
test_cutoff_date = pd.Timestamp(config["test_cutoff_date"])
train_index = [
    i for i, d in enumerate(data["temporal_cutoff"]) if pd.Timestamp(d) <= cutoff_date
]
test_index = [
    i
    for i, d in enumerate(data["temporal_cutoff"])
    if pd.Timestamp(d) > cutoff_date and pd.Timestamp(d) <= test_cutoff_date
]


print(f"Train size: {len(train_index)}")
print(f"Test size: {len(test_index)}")

from torch.utils.data import Dataset, DataLoader
from ast import literal_eval


def get_ct(bp, s):
    ct_matrix = np.zeros((len(s), len(s)))
    for b in bp:
        ct_matrix[b[0] - 1, b[1] - 1] = 1
    return ct_matrix


class RNA3D_Dataset(Dataset):
    def __init__(self, indices, data):
        self.indices = indices
        self.data = data
        self.tokens = {nt: i for i, nt in enumerate("ACGU")}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        sequence = [self.tokens[nt] for nt in (self.data["sequence"][idx])]
        sequence = np.array(sequence)
        sequence = torch.tensor(sequence)

        # get C1' xyz
        xyz = self.data["xyz"][idx]
        xyz = torch.tensor(np.array(xyz))

        if len(sequence) > config["max_len"]:
            crop_start = np.random.randint(len(sequence) - config["max_len"])
            crop_end = crop_start + config["max_len"]

            sequence = sequence[crop_start:crop_end]
            xyz = xyz[crop_start:crop_end]

        return {"sequence": sequence, "xyz": xyz}


train_dataset = RNA3D_Dataset(train_index, data)
val_dataset = RNA3D_Dataset(test_index, data)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


import sys

sys.path.append("/kaggle/input/8mexps-test4-wconfig")


from Network import *
import yaml


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)


def load_config_from_yaml(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout = 0.2
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(
                torch.load("models/epoch_0/pytorch_model_fsdp.bin", map_location="cpu")
            )
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1))
        self.dropout = nn.Dropout(0.2)
        self.xyz_predictor = nn.Linear(384, 3)

    def forward(self, src):
        # with torch.no_grad():
        sequence_features, pairwise_features = self.get_embeddings(
            src, torch.ones_like(src).long().to(src.device)
        )

        xyz = self.xyz_predictor(sequence_features)

        return xyz


model = finetuned_RibonanzaNet(
    load_config_from_yaml("configs/pairwise.yaml"), pretrained=True
).cuda()


def calculate_distance_matrix(X, Y, epsilon=1e-4):
    return (torch.square(X[:, None] - Y[None, :]) + epsilon).sum(-1).sqrt()


def dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()] = False

    if d_clamp is not None:
        rmsd = (torch.square(pred_dm[mask] - gt_dm[mask]) + epsilon).clip(0, d_clamp**2)
    else:
        rmsd = torch.square(pred_dm[mask] - gt_dm[mask]) + epsilon

    return rmsd.sqrt().mean() / Z


def local_dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=30):
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = (~torch.isnan(gt_dm)) * (gt_dm < d_clamp)
    mask[torch.eye(mask.shape[0]).bool()] = False

    rmsd = torch.square(pred_dm[mask] - gt_dm[mask]) + epsilon
    # rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).sqrt()/Z
    # rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])/Z
    return rmsd.sqrt().mean() / Z


def dRMAE(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()] = False

    rmsd = torch.abs(pred_dm[mask] - gt_dm[mask])

    return rmsd.mean() / Z


import torch


def align_svd_mae(input, target, Z=10):
    """
    Aligns the input (Nx3) to target (Nx3) using SVD-based Procrustes alignment
    and computes RMSD loss.

    Args:
        input (torch.Tensor): Nx3 tensor representing the input points.
        target (torch.Tensor): Nx3 tensor representing the target points.

    Returns:
        aligned_input (torch.Tensor): Nx3 aligned input.
        rmsd_loss (torch.Tensor): RMSD loss.
    """
    assert input.shape == target.shape, "Input and target must have the same shape"

    # mask
    mask = ~torch.isnan(target.sum(-1))

    input = input[mask]
    target = target[mask]

    # Compute centroids
    centroid_input = input.mean(dim=0, keepdim=True)
    centroid_target = target.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input - centroid_input.detach()
    target_centered = target - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)

    # Compute rotation matrix
    R = Vt @ U.T

    # Ensure a proper rotation (det(R) = 1, no reflection)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt @ U.T

    # Rotate input
    aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()

    # # Compute RMSD loss
    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())

    # rmsd_loss = torch.sqrt(((aligned_input - target) ** 2).mean())

    # return aligned_input, rmsd_loss
    return torch.abs(aligned_input - target).mean() / Z


from tqdm import tqdm
from torch.amp import GradScaler

epochs = 10
cos_epoch = 5


best_loss = np.inf
optimizer = torch.optim.Adam(
    model.parameters(), weight_decay=0, lr=0.00001
)  # no weight decay following AF

batch_size = 1

# for cycle in range(2):

criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

scaler = GradScaler()

schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=(epochs - cos_epoch) * len(train_loader) // batch_size
)

best_val_loss = 99999999999
for epoch in range(epochs):
    model.train()
    tbar = tqdm(train_loader)
    total_loss = 0
    oom = 0
    for idx, batch in enumerate(tbar):
        # try:
        sequence = batch["sequence"].cuda()
        gt_xyz = batch["xyz"].cuda().squeeze()

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        pred_xyz = model(sequence).squeeze()

        loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz) + align_svd_mae(
            pred_xyz, gt_xyz
        )
        # local_dRMSD(pred_xyz,pred_xyz,gt_xyz,gt_xyz)

        # if loss!=loss:
        #     stop

        (loss / batch_size).backward()

        if (idx + 1) % batch_size == 0 or idx + 1 == len(tbar):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            # scaler.scale(loss/batch_size).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # scaler.step(optimizer)
            # scaler.update()

            if (epoch + 1) > cos_epoch:
                schedule.step()
        # schedule.step()
        total_loss += loss.item()

        tbar.set_description(
            f"Epoch {epoch + 1} Loss: {total_loss / (idx + 1)} OOMs: {oom}"
        )

        # except Exception:
        #     #print(Exception)
        #     oom+=1
    tbar = tqdm(val_loader)
    model.eval()
    val_preds = []
    val_loss = 0
    for idx, batch in enumerate(tbar):
        sequence = batch["sequence"].cuda()
        gt_xyz = batch["xyz"].cuda().squeeze()

        with torch.no_grad():
            pred_xyz = model(sequence).squeeze()
            loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz)

        val_loss += loss.item()
        val_preds.append([gt_xyz.cpu().numpy(), pred_xyz.cpu().numpy()])
    val_loss = val_loss / len(tbar)
    print(f"val loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_preds = val_preds
        torch.save(model.state_dict(), "RibonanzaNet-3D.pt")

    # 1.053595052265986 train loss after epoch 0
torch.save(model.state_dict(), "RibonanzaNet-3D-final.pt")
