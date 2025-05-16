import os
import time
import argparse

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import RNADataset, Custom_Collate_Obj
from Network import RibonanzaNet
from Functions import load_and_split_rn2_ABCD, load_config_from_yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
parser.add_argument('--compile', action='store_true')
args = parser.parse_args()

np.random.seed(0)

config = load_config_from_yaml(args.config_path)
config.print()

# Reduce the number of training and validation samples for performance testing
hdf_files, train_indices, val_indices = load_and_split_rn2_ABCD(config)
train_indices = train_indices[:20]
val_indices = val_indices[:10]

print(f"train shape: {len(train_indices)}")
print(f"val shape: {len(val_indices)}")


seq_length = config.max_len

train_dataset = RNADataset(
    hdf_files,
    train_indices,
    flip=config.use_flip_aug,
    add_noise=config.use_noise_aug
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=Custom_Collate_Obj(config.max_len),
    pin_memory=True
)


val_dataset=RNADataset(hdf_files, val_indices, train=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.test_batch_size,
    shuffle=False,
    collate_fn=Custom_Collate_Obj(config.max_len),
)

model = RibonanzaNet(config).cuda()
if args.compile:
    print("compiling model")
    model = torch.compile(model, dynamic=False)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

criterion=torch.nn.L1Loss(reduction='none')

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    # training loop
    tbar = tqdm(train_loader)
    model.train()
    start = time.time()

    for idx, batch in enumerate(tbar):
        src = batch['sequence'].cuda()
        masks = batch['masks'].bool().cuda()
        labels = batch['labels'].cuda()
        SN = batch['SN'].cuda()

        loss_masks = batch['loss_masks'].cuda()
        SN = SN.reshape(SN.shape[0], 1, SN.shape[1]) >= 0.5
        loss_masks = loss_masks * SN

        SN=batch['SN'].cuda()
        output = model(src, masks)
        loss = criterion(output, labels)
        loss = loss * SN[:, None, :].clip(0.5, 1)
        loss = loss[loss_masks]
        loss = loss.mean()

        loss.backward()

        tbar.write(f"train loss: {loss.item():.4f}")

    print(f"training time: {time.time() - start:.4f}s")

    # validation loop
    model.eval()
    tbar = tqdm(val_loader)
    start = time.time()
    print("doing val")

    for idx, batch in enumerate(tbar):
        src = batch['sequence'].cuda()
        masks = batch['masks'].bool().cuda()
        labels = batch['labels'].cuda()
        loss_masks = batch['loss_masks'].cuda()

        with torch.no_grad():
            output = model(src, masks)
        loss = criterion(output, labels)[loss_masks]

        tbar.write(f"val loss: {loss.mean():.4f}")

    print(f"validation time: {time.time() - start:.4f}s")


# collect the table string
stats = prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total")

# prepare output path with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs("./prof", exist_ok=True)
outfile = f"./prof/profile_{timestamp}.txt"

# write to file
with open(outfile, "w", encoding='utf8') as f:
    f.write(stats)

print(f"Profile saved to {outfile}")
