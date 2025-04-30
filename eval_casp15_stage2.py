import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from utils import *
import os
from Diffusion import Diffusion

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
    "structural_violation_epoch": 50,
    "balance_weight": False,
}

test_data=pd.read_csv("../input/test_sequences.csv")

from torch.utils.data import Dataset, DataLoader

class RNADataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)




        return {'sequence':sequence}

test_dataset=RNADataset(test_data)



import sys

#sys.path.append("/kaggle/input/ribonanzanet2d-final")

import torch.nn as nn
from Network import finetuned_RibonanzaNet





model=finetuned_RibonanzaNet(load_config_from_yaml("pairwise.yaml"),pretrained=True).cuda()

diffusion = Diffusion(model,1000).cuda()
#model.decode(torch.ones(1,10).long().cuda(),torch.ones(1,10).long().cuda())


import torch
state_dict=torch.load("RibonanzaNet-3D-stage2-final-v2.pt",map_location='cpu')

#get rid of module. from ddp state dict
new_state_dict={}

for key in state_dict:
    new_state_dict[key[7:]]=state_dict[key]

model.load_state_dict(new_state_dict)

solution=pd.read_csv("../input/validation_labels.csv")

os.system('mkdir casp15_distograms')

from tqdm import tqdm
model.eval()
preds=[]
for i in tqdm(range(len(test_dataset))):
    src=test_dataset[i]['sequence'].long()
    src=src.unsqueeze(0).cuda()
    target_id=test_data.loc[i,'target_id']
    target_solution=solution[solution['ID'].str.contains(target_id)]
    gt_xyz=target_solution[['x_1','y_1','z_1']].values
    gt_distogram=calculate_distance_matrix(torch.tensor(gt_xyz),torch.tensor(gt_xyz)).numpy().clip(2,39)
    #model.eval()

    #tmp=[]
    predicted_dm=[]
    #for _ in range(5):
    with torch.no_grad():
        xyz,distogram=diffusion.sample(src,5)
        #xyz=xyz.squeeze()
    
    predicted_dm=[]
    for j in range(2):
        predicted_dm.append(calculate_distance_matrix(xyz[j],xyz[j]).cpu().numpy().clip(2,39))

        #tmp.append(xyz.cpu().numpy())
    

    plt.subplot(2,2,1)
    plt.imshow(gt_distogram,cmap='hot',interpolation='nearest')
    plt.title('ground truth distogram')
    #plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(distogram.cpu().numpy(),cmap='hot',interpolation='nearest')
    plt.title('predicted distogram')
    plt.subplot(2,2,3)
    plt.imshow(predicted_dm[0],cmap='hot',interpolation='nearest')
    plt.title('predicted structure distogram')
    plt.subplot(2,2,4)
    plt.imshow(predicted_dm[1],cmap='hot',interpolation='nearest')
    plt.title('predicted structure distogram')
    #plt.colorbar()
    plt.savefig(f"casp15_distograms/{target_id}.png")
    plt.clf()
    #exit()

    # model.eval()
    # with torch.no_grad():
    #     xyz=model(src)[-1].squeeze()
    #tmp.append(xyz.cpu().numpy())

    #tmp=np.stack(tmp,0)
    #exit()
    preds.append(xyz.cpu().numpy())


ID=[]
resname=[]
resid=[]
x=[]
y=[]
z=[]

data=[]

for i in range(len(test_data)):
    #print(test_data.loc[i])

    
    for j in range(len(test_data.loc[i,'sequence'])):
        # ID.append(test_data.loc[i,'sequence_id']+f"_{j+1}")
        # resname.append(test_data.loc[i,'sequence'][j])
        # resid.append(j+1) # 1 indexed
        row=[test_data.loc[i,'target_id']+f"_{j+1}",
             test_data.loc[i,'sequence'][j],
             j+1]

        for k in range(5):
            for kk in range(3):
                row.append(preds[i][k][j][kk])
        data.append(row)

columns=['ID','resname','resid']
for i in range(1,6):
    columns+=[f"x_{i}"]
    columns+=[f"y_{i}"]
    columns+=[f"z_{i}"]


submission=pd.DataFrame(data,columns=columns)


submission
submission.to_csv('submission.csv',index=False)

#score val
import pandas as pd
import pandas.api.types
import os
import re

# Function to parse TMscore output
def parse_tmscore_output(output):
    result = {}

    # Extract TM-score based on length of reference structure (second)
    tm_score_match = re.findall(r"TM-score=\s+([\d.]+)", output)[1]
    result['TM-score'] = float(tm_score_match) if tm_score_match else None

    return result

def write_pdb_line(atom_name, atom_serial, residue_name, chain_id, residue_num, x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P'):
    """
    Writes a single line of PDB format based on provided atom information. 
    
    Args:
        atom_name (str): Name of the atom (e.g., "N", "CA").
        atom_serial (int): Atom serial number.
        residue_name (str): Residue name (e.g., "ALA"). 
        chain_id (str): Chain identifier. 
        residue_num (int): Residue number. 
        x_coord (float): X coordinate.
        y_coord (float): Y coordinate.
        z_coord (float): Z coordinate.
        occupancy (float, optional): Occupancy value (default: 1.0). 
        b_factor (float, optional): B-factor value (default: 0.0). 
    
    Returns:
        str: A single line of PDB string.
    """
    line = f"ATOM  {atom_serial:>5d}  {atom_name:<5s} {residue_name:<3s} {residue_num:>3d}    {x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n"
    return line

def write2pdb(df, xyz_id, pdb_path):
    resolved_cnt=0
    with open(pdb_path, "w") as pdb_file:
        for _, row in df.iterrows():
            x_coord=row[f"x_{xyz_id}"]
            y_coord=row[f"y_{xyz_id}"]
            z_coord=row[f"z_{xyz_id}"]

            if x_coord>-1e17 and y_coord>-1e17 and z_coord>-1e17:
            #if True:
                resolved_cnt+=1
                pdb_line = write_pdb_line(
                    atom_name="C1'", 
                    atom_serial=int(row["resid"]), 
                    residue_name=row['resname'], 
                    chain_id='0', 
                    residue_num=int(row["resid"]), 
                    x_coord=x_coord, 
                    y_coord=y_coord, 
                    z_coord=z_coord,
                    atom_type="C"
                )
                pdb_file.write(pdb_line)
    return resolved_cnt


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Computes the TM-score between predicted and native RNA structures using USalign.

    This function evaluates the structural similarity of RNA predictions to native structures
    by computing the TM-score. It uses USalign, a structural alignment tool, to compare
    the predicted structures with the native structures.

    Workflow:
    1. Copies the USalign binary to the working directory and grants execution permissions.
    2. Extracts the `pdb_id` from the `ID` column of both the solution and submission DataFrames.
    3. Iterates over each unique `pdb_id`, grouping the native and predicted structures.
    4. Writes PDB files for native and predicted structures.
    5. Runs USalign on each predicted-native pair and extracts the TM-score.
    6. Computes the highest TM-score per target and returns aggregated results.

    Args:
        solution (pd.DataFrame): A DataFrame containing the native RNA structures.
        submission (pd.DataFrame): A DataFrame containing the predicted RNA structures.
        row_id_column_name (str): The name of the column containing unique row identifiers.

    Returns:
        tuple:
            - results (list): The highest TM-score for each `pdb_id`.
            - results_per_sub (list): TM-scores for each predicted-native pair.
            - outputs (list): Raw output logs from USalign for debugging.
    '''

    #os.system("cp //kaggle/input/usalign/USalign /kaggle/working/")
    #os.system("sudo chmod u+x /kaggle/working//USalign")


    # Extract pdb_id from ID (pdb_resid)
    solution["pdb_id"] = solution["ID"].apply(lambda x: x.split("_")[0])
    submission["pdb_id"] = submission["ID"].apply(lambda x: x.split("_")[0])

    #fix pdb_ids comment out later
    # solution.loc[solution['pdb_id']=="R1138v1",'pdb_id']='R1138'
    # solution.loc[solution['pdb_id']=="R1117",'pdb_id']='R1117v2'
    
    results=[]
    outputs=[]
    results_per_sub=[]
    # Iterate through each pdb_id and generate PDB files for both clean and corrupted data
    for pdb_id, group_native in solution.groupby("pdb_id"):
        group_predicted = submission[submission["pdb_id"] == pdb_id]
        #print(group_native,group_predicted)
        # Define output file paths
        # clean_pdb_path = os.path.join(output_folder, f"{pdb_id}_C3_clean.pdb")
        # corrupted_pdb_path = os.path.join(output_folder, f"{pdb_id}_C3_corrupted.pdb")
        native_pdb=f'native.pdb'
        predicted_pdb=f'predicted.pdb'

        all_scores=[]
        for pred_cnt in range(1,6):
            tmp=[]
            for native_cnt in range(1,41):
                # Write solution PDB
                resolved_cnt=write2pdb(group_native, native_cnt, native_pdb)
                
                # Write predicted PDB
                _=write2pdb(group_predicted, pred_cnt, predicted_pdb)

                if resolved_cnt>0:
                    command = f"../USalign {predicted_pdb} {native_pdb} -atom \" C1'\""
                    output = os.popen(command).read()
                    outputs.append(output)
                    parsed_data = parse_tmscore_output(output)
                    tmp.append(parsed_data['TM-score'])
                    
            all_scores.append(max(tmp))
        # print(output)
        # stop
        print(pdb_id)
        print(all_scores)
        results_per_sub.append(all_scores)
        results.append(max(all_scores))
    
    print(results)
    #return sum(results)/len(results), outputs
    return results, results_per_sub, outputs
    #return outputs

solution=pd.read_csv("../input/validation_labels.csv")

scores,results_per_sub,outputs=score(solution,submission,'ID')
print(np.mean(scores))