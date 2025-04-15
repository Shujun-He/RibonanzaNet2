from Dataset import *
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from pytorch_ranger import Ranger
import argparse
from accelerate import Accelerator
import time
import json
import matplotlib.pyplot as plt
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig
from accelerate.utils.fsdp_utils import save_fsdp_model
import multiprocessing
import h5py
from load_rawread import load_rawread, get_rawread_data

#multiprocessing.set_start_method("spawn")

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

#from torch.cuda.amp import GradScaler
#from torch import autocast
#if __name__ == "__main__":
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
parser.add_argument('--compile', type=str, default="true")

args = parser.parse_args()

np.random.seed(0)

config = load_config_from_yaml(args.config_path)
config.print()

accelerator = Accelerator(mixed_precision='bf16')

os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_id)
os.environ["TORCH_DISTRIBUTED_DEBUG"]='DETAIL'


os.system('mkdir logs')
os.system('mkdir models')
os.system('mkdir oofs')
logger=CSVLogger(['epoch','train_loss','train_rawread_loss','train_binary_rawread_loss','train_outer_product_loss','train_r_norm_loss',
                    'val_loss','val_rawread_loss','val_binary_rawread_loss','val_outer_product_loss','val_r_norm_loss'],
                    f'logs/fold{config.fold}.csv')
                  




all_data, train_indices, val_indices = get_rawread_data(config)



print(f"train shape: {len(train_indices)}")
print(f"val shape: {len(val_indices)}")



#pl_train=pl.read_parquet()
seq_length=256

# print(seq_length)
# exit()
num_workers = min(config.batch_size, multiprocessing.cpu_count() // 8)

train_dataset=RawReadRNADataset(train_indices,all_data,max_len=config.max_len,max_seq=config.max_seq)
train_loader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=4)

sample=train_dataset[0]

#exit()

val_dataset=RawReadRNADataset(val_indices,all_data,max_len=config.max_len)
val_loader=DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False,
                        num_workers=min(config.batch_size,16))

os.system('mkdir outer_product_plots')
for idx, batch in enumerate(train_loader):
    outer_products=batch['outer_products']
    outer_products=outer_products.numpy()
    #for b in range(len(outer_products)):
    #outer_product[outer_product!=outer_product]=0
    for j in range(2):
        outer_product=outer_products[0,:,:,j]
        plt.subplot(1,2,j+1)
        plt.imshow(outer_product,vmin=-1,vmax=1)
        #plt.colorbar()
    plt.savefig(f'outer_product_plots/{idx}.png')
    plt.clf()
    plt.close()
    if idx>10:
        break