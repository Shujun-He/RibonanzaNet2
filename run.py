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


hdf_files, train_indices, val_indices = load_and_split_rn2_ABCD(config)


# for data scaling experiments
if config.use_data_percentage<1:
    train_indices = train_indices[:int(len(train_indices)*config.use_data_percentage)]
    val_indices = val_indices[:int(len(val_indices)*config.use_data_percentage)]

    print(f"using {config.use_data_percentage} of data")
    print(f"train shape: {len(train_indices)}")
    print(f"val shape: {len(val_indices)}")

#exit()

# plot_and_save_bar_chart([data_dict['dataset_name'][i] for i in train_indices],
#                 f"dataset_cnt.png")


if hasattr(config,"dataset2drop"):
    print(f"dropping {config.dataset2drop} from training data")
    
    # print(set(data_dict['dataset_name']))
    # exit()
    train_indices=dataset_dropout(data_dict['dataset_name'], train_indices, config.dataset2drop)


print(f"train shape: {len(train_indices)}")
print(f"val shape: {len(val_indices)}")



#pl_train=pl.read_parquet()
seq_length=config.max_len

# print(seq_length)
# exit()
num_workers = min(config.batch_size, multiprocessing.cpu_count() // 8)

train_dataset=RNADataset(hdf_files,train_indices,
                        flip=config.use_flip_aug,
                        add_noise=config.use_noise_aug)
train_loader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,
                        collate_fn=Custom_Collate_Obj(config.max_len),num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=4)

# for i in range(10):
#     sample=train_dataset[i]
# exit()


val_dataset=RNADataset(hdf_files,val_indices,train=False)
val_loader=DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False,
                        collate_fn=Custom_Collate_Obj(config.max_len),num_workers=min(config.batch_size,16))


print(accelerator.distributed_type)


model=RibonanzaNet(config)#.cuda()

#model.load_state_dict(torch.load("../../exps/test41_biglr/models/epoch_14/pytorch_model_fsdp.bin",map_location='cpu'))

if config.previous_model_path != "none":
    load_state_dict_ignore_shape(model, config.previous_model_path)

#reinit last linear layer
#model.decoder.reset_parameters()

model=accelerator.prepare(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

optimizer = Ranger(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)
#optimizer = torch.optim.Adam(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)





criterion=torch.nn.L1Loss(reduction='none')
val_criterion=torch.nn.L1Loss(reduction='none')

#.to(accelerator.device)#.cuda().float()

cos_epoch=-1
print(f"cosine annealing from epoch {cos_epoch+1}")
lr_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,len(train_loader)//config.gradient_accumulation_steps)

# warmup_schduler=LinearWarmupScheduler(optimizer=optimizer,
#                                     total_steps=len(train_loader),
#                                     final_lr=config.learning_rate)
#exit()
optimizer, train_loader, val_loader, lr_schedule = accelerator.prepare(optimizer, train_loader, val_loader, lr_schedule)

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
    model.train()
    #for batch in tqdm(train_loader):

    for idx, batch in enumerate(tbar):
        
        src=batch['sequence']#.cuda()
        masks=batch['masks'].bool()#.cuda()
        labels=batch['labels']#.cuda()
        SN=batch['SN']
        #print_learning_rates(optimizer)
        

        bs=len(labels)
        #batch_attention_mask=batch['attention_mask'].unsqueeze(1)[:,:,:src.shape[-1],:src.shape[-1]]

        loss_masks=batch['loss_masks']#.cuda()
#SSH FS test 
        SN=SN.reshape(SN.shape[0],1,SN.shape[1])>=0.5
        loss_masks=loss_masks*SN

        # print(SN.shape)
        # print(loss_masks.shape)
        # exit()

        #exit()
        #batch_attention_mask=batch['attention_mask']
        #batch_attention_mask=torch.stack([batch_attention_mask[:,:src.shape[-1],:src.shape[-1]],bpp],1)
        SN=batch['SN']
        # print(SN.shape)
        # exit()
        with accelerator.autocast():
            output=model(src,masks)
            loss=criterion(output,labels)#*loss_weight BxLxC
            loss=loss*SN[:,None,:].clip(0.5,1) #weight with SN, downweight low quality data, and high quality data has up to 1 weight
            loss=loss[loss_masks]
            loss=loss.mean()
            #exit()
        #exit()
        accelerator.backward(loss/config.gradient_accumulation_steps)
        
        #loss.backward()
        if (idx + 1) % config.gradient_accumulation_steps == 0:
            total_steps+=1
            #if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            #optimizer.step()
            optimizer_step()
            optimizer.zero_grad()
            if epoch > cos_epoch:
                lr_schedule.step()
            # elif epoch == 0:
            #     warmup_schduler.step()

        
            if total_steps % config.log_interval == 0:
                accelerator.save_state(f"models/step_{total_steps}",safe_serialization=False)


        total_loss+=loss.item()
        #exit()
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)}")
        

        #break
    train_loss=total_loss/(idx+1)
    # if accelerator.is_local_main_process:
    #     if epoch==cos_epoch:
    #     #     torch.save(accelerator.unwrap_model(model).state_dict(),f"models/model{config.fold}_pl_only.pt")
    #     # torch.save(accelerator.unwrap_model(optimizer).state_dict(),f"models/optimizer{config.fold}.pt")
    #         accelerator.save_state("cos_models")

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
        masks=batch['masks'].bool()#.cuda()
        labels=batch['labels']#.cuda()
        bs=len(labels)
        loss_masks=batch['loss_masks']#.cuda()
        src_flipped=src.clone()

        length=batch['length']
        for batch_idx in range(len(src)):
            src_flipped[batch_idx,:length[batch_idx]]=src_flipped[batch_idx,:length[batch_idx]].flip(0)
        src_flipped=src_flipped.clone()

        #with accelerator.autocast():
        with torch.no_grad():
            with accelerator.autocast():
                output=model(src,masks)
                if config.use_flip_aug:
                    flipped_output=model(src_flipped,masks)
                    for batch_idx in range(len(flipped_output)):
                        flipped_output[batch_idx,:length[batch_idx]]=flipped_output[batch_idx,:length[batch_idx]].flip(0)

                    output=(flipped_output+output)/2
        loss=val_criterion(output,labels)[loss_masks]

        L=src.shape[1]
        to_pad=seq_length-L
        #output=output#[loss_masks]
        #labels=labels#[loss_masks]

        output=F.pad(output,(0,0,0,to_pad),value=0)
        labels=F.pad(labels,(0,0,0,to_pad),value=0)
        loss_masks=F.pad(loss_masks,(0,0,0,to_pad),value=0)

        all_output = accelerator.gather(output)
        all_labels = accelerator.gather(labels)
        all_masks = accelerator.gather(loss_masks)

        preds.append(all_output)
        gts.append(all_labels)
        val_loss_masks.append(all_masks)

        loss=loss.mean()
        #loss=torch.sqrt(loss)
        val_loss+=loss.item()

        tbar.set_description(f"Epoch {epoch + 1} Val Loss: {val_loss/(idx+1)}")

        

    #val_loss=val_loss/len(tbar)
        #break
    preds=torch.cat(preds)
    gts=torch.cat(gts)
    val_loss_masks=torch.cat(val_loss_masks)


    # print(accelerator.is_main_process)
    # exit()
    if accelerator.is_main_process:
        val_loss=val_criterion(preds[val_loss_masks],gts[val_loss_masks]).mean().item()

        logger.log([epoch,train_loss,val_loss])

        if val_loss<best_val_loss:
            best_val_loss=val_loss
            data_dict = {
                            "preds": preds.cpu().numpy(),
                            "gts": gts.cpu().numpy(),
                            "val_loss_masks": val_loss_masks.cpu().numpy()
                        }

            # Save to pickle file
            with open(f"oofs/{config.fold}.pkl", "wb+") as file:
                pickle.dump(data_dict, file)
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
