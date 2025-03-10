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
from scipy.stats import pearsonr

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
logger=CSVLogger(['epoch','train_loss','val_loss'],f'logs/fold{config.fold}.csv')


# with open('data/data_dict.p','rb') as f:
#     data_dict=pickle.load(f)
#     print(data_dict.keys())

# List of prefixes
prefixes = [
    "RTB000_Marathon_Bicine_3pct_DMS",
    "RTB001_Marathon_Bicine_nomod",
    "RTB002_SSII_Bicine_2A3",
    "RTB003_SSII_Bicine_nomod"
]

lengths=[606861064,276294348,546241823,258796672]


input_dir = '/lustre/fs0/scratch/shujun/BothLanes_RPT'
min_read = 128
#create filter vector where num_reads>min_read for all prefixes
filter_vector=[]
for prefix, length in zip(prefixes, lengths):
    csv=pl.read_csv(f'{input_dir}/{prefix}.index.csv')
    filter_vector.append(csv["num_reads"].to_numpy()>min_read)

filter_vector=np.stack(filter_vector,-1).sum(-1)==len(prefixes)
#exit()
train_labels=pl.read_csv("../../input/train_data.csv")
train_labels=train_labels[:len(train_labels)//2].unique(subset=['sequence'])

first_seq="GGGAACGACUCGAGUAGAGUCGAAAAUUUCCUUCCAAAUCCUGAGGGAGAGAUAGAGGCGGAGGGUCUGGGGGAGGAAUUAAAACACAAGGUCUCCUCCCCUCUCGCCUGUCCGAACUUGGGGGCACCCCGGCUCGUACUUCGGUACGAGCCGGGGAAAAGAAACAACAACAACAAC"



data={}
for prefix, length in zip(prefixes, lengths):

    raw_data = np.memmap(f'{input_dir}/{prefix}.reads.txt.memmap', dtype=np.uint8, mode='r', shape=(length, 192))
    csv=pl.read_csv(f'{input_dir}/{prefix}.index.csv')#[:1000]
    csv=csv.filter(filter_vector)
    rawread_indices=[[i,j] for i,j in zip(csv['read_start'].to_numpy(),csv['read_end'].to_numpy())]
    sequences=csv['sequence'].to_list()

    data[prefix]={'raw_data':raw_data,'rawread_indices':rawread_indices,'sequences':sequences}

labels_2a3=csv.join(train_labels,on='sequence',how='left')

r_norm=labels_2a3[[f'reactivity_{i+1:04d}' for i in range(177)]].to_numpy()

#exit()

#exit()

#StratifiedKFold on dataset
kfold=KFold(n_splits=config.nfolds,shuffle=True, random_state=0)
fold_indices={}
# dataset_name=csv['dataset_name'].to_numpy()
for i, (train_index, test_index) in enumerate(kfold.split(np.arange(len(csv)))):
    fold_indices[i]=train_index, test_index
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

# if config.use_dirty_data:
#     print(f"number of sequences in train {len(train_indices)}")
#     train_indices=np.concatenate([train_indices,dirty_data_indices])
#     print(f"number of sequences in train {len(train_indices)} after using dirty data")

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

train_dataset=RawReadRNADataset(train_indices,data,max_len=config.max_len,max_seq=config.max_seq)
train_loader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=4)

sample=train_dataset[0]

#exit()

val_dataset=RawReadRNADataset(val_indices,data,max_len=config.max_len)
val_loader=DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False,
                        num_workers=min(config.batch_size,16))

#exit()


print(accelerator.distributed_type)


model=RibonanzaNet(config)#.cuda()

model=accelerator.prepare(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

optimizer = Ranger(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)
#optimizer = torch.optim.Adam(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)

model.load_state_dict(torch.load("models/epoch_9/pytorch_model_fsdp.bin"))



criterion=torch.nn.CrossEntropyLoss(reduction='none')
val_criterion=torch.nn.CrossEntropyLoss(reduction='none')

#.to(accelerator.device)#.cuda().float()

cos_epoch=int(config.epochs*0.75)-1
lr_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(config.epochs-cos_epoch)*len(train_loader)//config.gradient_accumulation_steps)

warmup_schduler=LinearWarmupScheduler(optimizer=optimizer,
                                    total_steps=len(train_loader),
                                    final_lr=config.learning_rate)
#exit()
optimizer, train_loader, val_loader, lr_schedule, warmup_schduler= accelerator.prepare(optimizer, train_loader, val_loader, lr_schedule, warmup_schduler)

@torch.compile(fullgraph=False)
def optimizer_step():
    optimizer.step()
    
if args.compile == 'true':
    model = torch.compile(model,dynamic=False)
#model = model

best_val_loss=np.inf
for epoch in range(config.epochs):

    # training loop
    
    tbar = tqdm(train_loader)
    total_loss=0
    model.train()
    #for batch in tqdm(train_loader):

    for idx, batch in enumerate(tbar):
        
        src=batch['sequence']#.cuda()
        masks=batch['mask'].bool()#.cuda()
        rawreads=batch['rawreads']#.cuda()
        rawreads_mask=batch['rawreads_mask']#.cuda()
        labels=rawreads[:,:,1:]

        #print(labels.max())

        bs=len(src)
        #exit()
        # with accelerator.autocast():
        #     output=model(src,rawreads[:,:,:-1],masks)
        #     output=output.permute(0,3,1,2)
        #     loss=criterion(output,labels)#*loss_weight BxLxC
        #     loss_masks=rawreads_mask[:,:,1:].bool()
        #     loss=loss[loss_masks]
        #     loss=loss.mean()
            # optimizer.zero_grad()
            # continue
        #exit()
        tgt=torch.cat([torch.ones(1,1).cuda()*8,src],-1)[:,:-1].long()
        tgt2=torch.cat([torch.ones(1,1).cuda()*9,src],-1)[:,:-1].long()
        output=model(src,tgt.unsqueeze(1),masks).softmax(-1).squeeze(1)-model(src,tgt2.unsqueeze(1),masks).softmax(-1).squeeze(1)
        #output[0,0]

        self_prob=[]
        for i in range(tgt.size(1)):
            nt=src[0,i].item()
            self_prob.append(1-output[0,i,nt].item())
        self_prob=np.array(self_prob)

        r=r_norm[train_indices[idx]].astype('float32').clip(0,1)
        
        

        #plt.plot((self_prob-self_prob[25:140].mean())/self_prob[25:140].std())
        try:
            corr=pearsonr(self_prob[26:125],r[26:125])[0]
        except:
            corr=0
        z=(self_prob-self_prob[26:125].mean())/self_prob[26:125].std()
        #plt.plot(z[26:125],label="z of self_prob")
        plt.plot(self_prob[26:125],label="self_prob")
        plt.plot(r[26:125],label="r")
        plt.legend()
        plt.title(f"Correlation: {corr}")
        plt.savefig(f"plots/self_prob_{epoch}_{idx}.png")
        plt.close()

        #exit()
        # accelerator.backward(loss/config.gradient_accumulation_steps)
        
        # #loss.backward()
        # if (idx + 1) % config.gradient_accumulation_steps == 0:
        #     #if accelerator.sync_gradients:
        #     accelerator.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        #     #optimizer.step()
        #     optimizer_step()
        #     optimizer.zero_grad()
        #     if epoch > cos_epoch:
        #         lr_schedule.step()
        #     elif epoch == 0:
        #         warmup_schduler.step()

        
        # total_loss+=loss.item()
        # #exit()
        # tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)}")
        
        if idx>30:
            exit()
        #break
    train_loss=total_loss/(idx+1)


    # validation loop
    model.eval()
    tbar = tqdm(val_loader)
    val_loss=0
    preds=[]
    gts=[]
    print("doing val")
    val_loss_masks=[]

    for idx,batch in enumerate(tbar):
        src=batch['sequence']#.cuda()
        masks=batch['mask'].bool()#.cuda()
        rawreads=batch['rawreads']#.cuda()
        rawreads_mask=batch['rawreads_mask']#.cuda()
        labels=rawreads[:,:,1:]


        #with accelerator.autocast():
        with torch.no_grad():
            with accelerator.autocast():
                output=model(src,rawreads[:,:,:-1],masks)
                output=output.permute(0,3,1,2)
                loss=criterion(output,labels)#*loss_weight BxLxC
                loss_masks=rawreads_mask[:,:,1:].bool()
                loss=loss[loss_masks]
                loss=loss.mean()


        loss = accelerator.gather(loss).mean()
        # all_labels = accelerator.gather(labels)
        # all_masks = accelerator.gather(loss_masks)

        val_loss+=loss.item()


        tbar.set_description(f"Epoch {epoch + 1} Val Loss: {val_loss/(idx+1)}")

        

    val_loss=val_loss/len(tbar)
        #break
    # preds=torch.cat(preds)
    # gts=torch.cat(gts)
    # val_loss_masks=torch.cat(val_loss_masks)


    # print(accelerator.is_main_process)
    # exit()
    if accelerator.is_main_process:
        #val_loss=val_criterion(preds[val_loss_masks],gts[val_loss_masks]).mean().item()

        logger.log([epoch,train_loss,val_loss])

        if val_loss<best_val_loss:
            best_val_loss=val_loss
            #if torch.distributed.get_rank() == 0:
            #torch.save(accelerator.unwrap_model(model).state_dict(),f"models/model{config.fold}.pt")
            #torch.save(model.state_dict(),f"models/model{config.fold}.pt")
            #state_dict=accelerator.get_state_dict(model)
            #torch.save(state_dict,f"models/model{config.fold}.pt")
            # print(accelerator.unwrap_model(model).state_dict())
            # unwrapped_model=accelerator.unwrap_model(model)
            # full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            #     state = accelerator.get_state_dict(unwrapped_model)
            # exit()
            # Using context manager for saving
            # with FSDP.state_dict_type(
            #     model, 
            #     StateDictType.SHARDED_STATE_DICT,
            #     ShardedStateDictConfig(offload_to_cpu=True)  # Optionally offload to CPU
            # ):
            #     state_dict = model.state_dict()
            #     # Each rank saves its own shard
            #     torch.save(state_dict, f"model-shard-{torch.distributed.get_rank()}.pt")

            #accelerator.save_model(model, f"models/model{config.fold}.pt")
            #accelerator.save_state("models")
            # data_dict = {
            #                 "preds": preds.cpu().numpy(),
            #                 "gts": gts.cpu().numpy(),
            #                 "val_loss_masks": val_loss_masks.cpu().numpy()
            #             }

            # # Save to pickle file
            # with open(f"oofs/{config.fold}.pkl", "wb+") as file:
            #     pickle.dump(data_dict, file)
    save_start_time=time.time()
    #if accelerator.is_main_process:
    accelerator.save_state(f"models/epoch_{epoch}",safe_serialization=False)
    #accelerator.save(model, f"models/epoch_{epoch}.pkl",safe_serialization=False)
    save_time=time.time()-save_start_time
    print(f"It took {save_time} secs to save weights")
    #save_fsdp_model()
    #save_fsdp_model(accelerator.state.fsdp_plugin, accelerator, model, output_dir, i)
    # if val_loss<best_val_loss:
    #     accelerator.save_state("ckpt")

    #exit()
    #exit()

if accelerator.is_main_process:
    #torch.save(accelerator.unwrap_model(model).state_dict(),f"models/model{config.fold}_lastepoch.pt")

    end_time = time.time()
    elapsed_time = end_time - start_time

    with open("run_stats.json", 'w') as file:
            json.dump({'Total_execution_time': elapsed_time}, file, indent=4)
