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
from load_rawread import load_rawread

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
                  


# with open('data/data_dict.p','rb') as f:
#     data_dict=pickle.load(f)
#     print(data_dict.keys())

# # List of prefixes
# prefixes = [
#     "RTB000_Marathon_Bicine_3pct_DMS",
#     "RTB002_SSII_Bicine_2A3",
# ]

# filter_prefixes = [
#     "RTB000_Marathon_Bicine_3pct_DMS",
#     "RTB002_SSII_Bicine_2A3",
# ]
# input_dir = '/lustre/fs0/scratch/shujun/BothLanes_RawReads'
# lengths=[599528185,496581631]

# hdf5_data=h5py.File(f'{input_dir}/OneMil.v0.1.0.hdf5','r')
# #exit()

# min_read = 128
# #create filter vector where num_reads>min_read for all prefixes
# # filter_vector=[]
# # for prefix, length in zip(filter_prefixes, lengths):
# #     csv=pl.read_csv(f'{input_dir}/{prefix}.index.csv')
# #     filter_vector.append(csv["num_reads"].to_numpy()>min_read)

# # #filter_vector=np.stack(filter_vector,-1).sum(-1)==len(prefixes)
# # filter_vector=np.stack(filter_vector,-1).sum(-1)>0
# filter_vector=hdf5_data['signal_to_noise'][:].max(-1)>1.0
# #exit()


# #exit()
# data={}
# for prefix, length in zip(prefixes, lengths):

#     raw_data = np.memmap(f'{input_dir}/{prefix}.align_reads.txt.memmap', dtype=np.uint8, mode='r', shape=(length, 177))
#     csv=pl.read_csv(f'{input_dir}/{prefix}.index.csv')#[:1000]
#     csv=csv.filter(filter_vector)
#     rawread_indices=[[i,j] for i,j in zip(csv['read_start'].to_numpy(),csv['read_end'].to_numpy())]
#     sequences=csv['sequence'].to_list()

#     data[prefix]={'raw_data':raw_data,'rawread_indices':rawread_indices,'sequences':sequences}

data = load_rawread()

hdf5_data=h5py.File(f'../../input/Ribonanza2A_Genscript.v0.1.0.hdf5','r')
#save rnorm to memmap
rnorm_data={}
rnorm_data['r_norm']=hdf5_data['r_norm']#[filter_vector]
#rnorm_data['r_norm_index']=np.arange(len(filter_vector))[filter_vector]
rnorm_data['signal_to_noise']=hdf5_data['signal_to_noise']
dataset_name=pl.read_csv("../../sublib_id.csv")['sublibrary'].to_list()
#exit()

#exit()

#get high snr data indices
snr=hdf5_data['signal_to_noise'][:]
high_quality_indices = np.where((snr>1.).sum(1)==2)[0]
dirty_data_indices = np.where(((snr>1.).sum(1)==1))[0]

#dataset names
sublib_data=pl.read_csv('../../sublib_id.csv')['sublibrary'].to_list()

#StratifiedKFold on dataset
kfold=StratifiedKFold(n_splits=config.nfolds,shuffle=True, random_state=0)
fold_indices={}
high_quality_dataname=[sublib_data[i] for i in high_quality_indices]
for i, (train_index, test_index) in enumerate(kfold.split(high_quality_indices, high_quality_dataname)):
    fold_indices[i]=(high_quality_indices[train_index],high_quality_indices[test_index])
#exit()

train_indices=fold_indices[config.fold][0]
val_indices=fold_indices[config.fold][1]
#exit()
# for data scaling experiments
if config.use_data_percentage<1:
    print(f"Only using {config.use_data_percentage:.02f} of data")
    size=int(config.use_data_percentage*len(train_indices))
    train_indices=np.random.choice(train_indices,size,replace=False)
    print(f"number of sequences in train {len(train_indices)} after subsampling")

if config.use_dirty_data:
    print(f"number of sequences in train {len(train_indices)}")
    train_indices=np.concatenate([train_indices,dirty_data_indices])
    print(f"number of sequences in train {len(train_indices)} after using dirty data")

#exit()

# plot_and_save_bar_chart([data_dict['dataset_name'][i] for i in train_indices],
#                 f"dataset_cnt.png")


if hasattr(config,"dataset2drop"):
    print(f"dropping {config.dataset2drop} from training data")
    
    # print(set(data_dict['dataset_name']))
    # exit()
    train_indices=dataset_dropout(data_dict['dataset_name'], train_indices, config.dataset2drop)


print(f"train shape: {train_indices.shape}")
print(f"val shape: {val_indices.shape}")



#pl_train=pl.read_parquet()
seq_length=256

# print(seq_length)
# exit()
num_workers = min(config.batch_size, multiprocessing.cpu_count() // 8)

train_dataset=RawReadRNADataset(train_indices,data,rnorm_data,max_len=config.max_len,max_seq=config.max_seq)
train_loader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=4)

sample=train_dataset[0]

#exit()

val_dataset=RawReadRNADataset(val_indices,data,rnorm_data,max_len=config.max_len)
val_loader=DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False,
                        num_workers=min(config.batch_size,16))

#exit()


print(accelerator.distributed_type)


model=RibonanzaNet(config)#.cuda()


if config.previous_checkpoint != 'none':
    load_state_dict_ignore_shape(model,config.previous_checkpoint)

model=accelerator.prepare(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

optimizer = Ranger(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)
#optimizer = torch.optim.Adam(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)





criterion=torch.nn.CrossEntropyLoss(reduction='none') #
binary_CE=torch.nn.BCEWithLogitsLoss(reduction='none')
val_criterion=torch.nn.CrossEntropyLoss(reduction='none')

#.to(accelerator.device)#.cuda().float()

cos_epoch=int(config.epochs*0.75)-1
lr_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(config.epochs-cos_epoch)*len(train_loader)//config.gradient_accumulation_steps)

# warmup_schduler=LinearWarmupScheduler(optimizer=optimizer,
#                                     total_steps=len(train_loader),
#                                     final_lr=config.learning_rate)
#exit()
optimizer, train_loader, val_loader, lr_schedule= accelerator.prepare(optimizer, train_loader, val_loader, lr_schedule)

@torch.compile(fullgraph=False)
def optimizer_step():
    optimizer.step()
    
if args.compile == 'true':
    model = torch.compile(model,dynamic=False)
#model = model

best_val_loss=np.inf
total_steps=0
for epoch in range(config.epochs):

    # training loop
    
    tbar = tqdm(train_loader)
    total_loss=0
    total_outer_product_loss=0
    total_binary_loss=0
    total_raw_read_loss=0
    total_r_norm_loss=0
    model.train()
    #for batch in tqdm(train_loader):
            # data={'sequence':sequence,
            #   "rawreads":rawreads,
            #   "outer_products":all_outer_products,
            #   "mask":mask,
            #   "rawreads_mask":rawreads_mask,
            #   "r_norm":r_norm,
            #   "snr":snr}
    for idx, batch in enumerate(tbar):
        
        # print(batch['sequence'].shape)
        # print(batch['mask'].shape)
        # print(batch['rawreads'].shape)
        # print(batch['rawreads_mask'].shape)
        # print(batch['r_norm'].shape)
        # print(batch['snr'].shape)


        # exit()
        
        #if any loss is nan, exit
        # if torch.isnan(raw_read_loss).any():
        #     print("raw read loss is nan")
        #     exit()
        if torch.isnan(batch['outer_products']).all():
            print("outer_products are all nan")
            exit()          
        # if torch.isnan(outer_product_loss).any():
        #     print("outer product loss is nan")
        #     exit()
        # if torch.isnan(r_norm_loss).any():
        #     print("r norm loss is nan")
        #     exit()
            


