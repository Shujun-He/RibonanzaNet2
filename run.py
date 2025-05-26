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

if config.use_data_percentage<1:
    train_indices = train_indices[:int(len(train_indices)*config.use_data_percentage)]
    val_indices = val_indices[:int(len(val_indices)*config.use_data_percentage)]

    print(f"using {config.use_data_percentage} of data")
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

warmup_schduler=LinearWarmupScheduler(optimizer=optimizer,
                                    total_steps=config.warmup_steps*accelerator.num_processes,
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
    #if config.use_gradient_checkpoint:
    model.set_global_attr("use_gradient_checkpoint",config.use_gradient_checkpoint)

    for idx, batch in enumerate(tbar):
        
        # print(batch['sequence'].shape)
        # print(batch['mask'].shape)
        # print(batch['rawreads'].shape)
        # print(batch['rawreads_mask'].shape)
        # print(batch['r_norm'].shape)
        # print(batch['snr'].shape)


        # exit()
        raw_read_loss, binary_loss, outer_product_loss, r_norm_loss = rawread_training_loss(batch, model, accelerator)
        
        #if any loss is nan, exit
        # if torch.isnan(raw_read_loss).any():
        #     print("raw read loss is nan")
        #     exit()
        # if torch.isnan(binary_loss).any():
        #     print("binary loss is nan")
        #     exit()          
        # if torch.isnan(outer_product_loss).any():
        #     print("outer product loss is nan")
        #     exit()
        # if torch.isnan(r_norm_loss).any():
        #     print("r norm loss is nan")
        #     exit()
            

        #weighted loss from config
        loss = config.raw_read_loss_weight * raw_read_loss +\
               config.binary_loss_weight * binary_loss + \
               config.outer_product_loss_weight * outer_product_loss + \
               config.r_norm_loss_weight * r_norm_loss
        
        #track all losses
        total_outer_product_loss+=outer_product_loss.item()
        total_raw_read_loss+=raw_read_loss.item()
        total_binary_loss+=binary_loss.item()
        total_r_norm_loss+=r_norm_loss.item()


        accelerator.backward(loss/config.gradient_accumulation_steps)
        
        #loss.backward()
        if (idx + 1) % config.gradient_accumulation_steps == 0:
            #if accelerator.sync_gradients:
            total_steps+=1
            accelerator.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            #optimizer.step()
            optimizer_step()
            optimizer.zero_grad()
            if epoch > cos_epoch:
                lr_schedule.step()
            # elif epoch == 0:
            #     warmup_schduler.step()
            if total_steps < config.warmup_steps:
                warmup_schduler.step()

        if total_steps % config.log_interval == 0:
            accelerator.save_state(f"models/step_{total_steps}",safe_serialization=False)

        total_loss+=loss.item()
        #exit()
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1):.2f} Outer Product Loss: \
{total_outer_product_loss/(idx+1):.2f} Raw Read Loss: {total_raw_read_loss/(idx+1):.2f}, \
Binary Loss: {total_binary_loss/(idx+1):.2f} R Norm Loss: {total_r_norm_loss/(idx+1):.2f}")
        

        #break

    print(f'learning rate at end of epoch {epoch}')
    print_learning_rates(optimizer)

    
    train_loss=total_loss/(idx+1)
    total_outer_product_loss=total_outer_product_loss/(idx+1)
    total_raw_read_loss=total_raw_read_loss/(idx+1)
    total_binary_loss=total_binary_loss/(idx+1)
    total_r_norm_loss=total_r_norm_loss/(idx+1)

    # validation loop
    model.eval()
    model.set_global_attr("use_gradient_checkpoint",False)
    tbar = tqdm(val_loader)
    val_loss=0
    total_val_outer_product_loss=0
    total_val_binary_loss=0
    total_val_raw_read_loss=0
    total_val_r_norm_loss=0


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
                raw_read_loss, binary_loss, outer_product_loss, r_norm_loss = rawread_training_loss(batch, model, accelerator)

                #weighted loss from config
                loss = config.raw_read_loss_weight * raw_read_loss +\
                    config.binary_loss_weight * binary_loss + \
                    config.outer_product_loss_weight * outer_product_loss + \
                    config.r_norm_loss_weight * r_norm_loss

        
        #gather losses from all processes
        loss = accelerator.gather(loss).mean()
        raw_read_loss = accelerator.gather(raw_read_loss).mean()
        binary_loss = accelerator.gather(binary_loss).mean()
        outer_product_loss = accelerator.gather(outer_product_loss).mean()
        r_norm_loss = accelerator.gather(r_norm_loss).mean()

        #track all losses
        total_val_outer_product_loss+=outer_product_loss.item()
        total_val_raw_read_loss+=raw_read_loss.item()
        total_val_binary_loss+=binary_loss.item()
        total_val_r_norm_loss+=r_norm_loss.item()
        val_loss+=loss.item()


        tbar.set_description(f"Epoch {epoch + 1} Val Loss: {val_loss/(idx+1):.2f} Outer Product Loss: \
{total_val_outer_product_loss/(idx+1):.2f} Raw Read Loss: {total_val_raw_read_loss/(idx+1):.2f}, \
Binary Loss: {total_val_binary_loss/(idx+1):.2f} R Norm Loss: {total_val_r_norm_loss/(idx+1):.2f}")

    

    val_loss=val_loss/len(tbar)
    total_val_outer_product_loss=total_val_outer_product_loss/len(tbar)
    total_val_raw_read_loss=total_val_raw_read_loss/len(tbar)
    total_val_binary_loss=total_val_binary_loss/len(tbar)
    total_val_r_norm_loss=total_val_r_norm_loss/len(tbar)



    # print(accelerator.is_main_process)
    # exit()
    if accelerator.is_main_process:
        #val_loss=val_criterion(preds[val_loss_masks],gts[val_loss_masks]).mean().item()

        logger.log([epoch,train_loss,total_raw_read_loss,total_binary_loss,total_outer_product_loss,total_r_norm_loss,
                    val_loss,total_val_raw_read_loss,total_val_binary_loss,total_val_outer_product_loss,total_val_r_norm_loss])

        if val_loss<best_val_loss:
            best_val_loss=val_loss

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
