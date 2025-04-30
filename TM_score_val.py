 #code to implement best of 5 tm in val
 #unused for now
 
  total_loss=total_loss/len(tbar)

    tbar=tqdm(val_loader)
    model.eval()
    val_preds=[]
    val_loss=0
    val_rmsd=0
    val_lddt=0
    val_tm_score=0
    #unwrapped_diffusion=accelerator.unwrap_model(diffusion)
    #unwrapped_model=accelerator.unwrap_model(model)
    for idx, batch in enumerate(tbar):
        sequence=batch['sequence'].cuda()
        gt_xyz=batch['xyz'].cuda().squeeze()

        with torch.no_grad():
            # if accelerator.dis
            #pred_xyz=model.module.decode(sequence,torch.ones_like(sequence).long().cuda()).squeeze()
            if accelerator.distributed_type=='NO':
                pred_xyz=model.sample(sequence,5)[0].squeeze(0)
            else:
                pred_xyz=model.module.sample(sequence,5)[0].squeeze(0)
            #pred_xyz=model(sequence)[-1].squeeze()
            losses_5=[]
            rmsd_5=[]
            lddt_5=[]
            for i in range(5):
                loss=dRMAE(pred_xyz[i],pred_xyz[i],gt_xyz,gt_xyz)
                lddt=compute_lddt(pred_xyz[i].cpu().numpy(),gt_xyz.cpu().numpy())
                rmsd=align_svd_rmsd(pred_xyz[i],gt_xyz)

                losses_5.append(loss.item())
                lddt_5.append(lddt)
                rmsd_5.append(rmsd.item())

        sequence= ['ACGU'[int(i)] for i in sequence.cpu().numpy().reshape(-1)]
        
        #score RibonanzaTM
        submission=pd.DataFrame()
        submission['ID'] = [f"{accelerator.process_index}_{i}" for i in range(pred_xyz.shape[1])]
        for pred_id in range(1,6):
            submission[f'x_{pred_id}'] = pred_xyz[pred_id-1,:,0].cpu().numpy()
            submission[f'y_{pred_id}'] = pred_xyz[pred_id-1,:,1].cpu().numpy()
            submission[f'z_{pred_id}'] = pred_xyz[pred_id-1,:,2].cpu().numpy()
        submission['resid']=np.arange(1,pred_xyz.shape[1]+1)
        submission['resname']=sequence

        solution=pd.DataFrame()
        solution['ID'] = [f"{accelerator.process_index}_{i}" for i in range(len(gt_xyz))]
        #for gt_id in range(1,6):
        gt_id=1
        solution[f'x_{gt_id}'] = gt_xyz[:,0].cpu().numpy()
        solution[f'y_{gt_id}'] = gt_xyz[:,1].cpu().numpy()
        solution[f'z_{gt_id}'] = gt_xyz[:,2].cpu().numpy()
        # for sol_id in range(2,41):
        #     solution[f'x_{sol_id}'] = -1e18
        #     solution[f'y_{sol_id}'] = -1e18
        #     solution[f'z_{sol_id}'] = -1e18
        # solution=solution.fillna(-1e18)
        solution['resid']=np.arange(1,pred_xyz.shape[1]+1)
        solution['resname']=sequence

        tm_score,_,_=score(solution,submission,'ID',1)


        val_rmsd+=accelerator.gather(torch.tensor(np.min(losses_5)).to(pred_xyz.device)).mean().item()
        val_lddt+=accelerator.gather(torch.tensor(np.max(lddt_5)).to(pred_xyz.device)).mean().item()
        val_loss+=accelerator.gather(torch.tensor(np.min(rmsd_5))).mean().item()
        val_tm_score+=accelerator.gather(torch.tensor(tm_score)).mean().item()
        exit()
        


        val_preds.append([gt_xyz.cpu().numpy(),pred_xyz.cpu().numpy()])