import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import numpy as np
import yaml
import pandas as pd
import os
import re

from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of steps in one epoch (len(train_loader))
        final_lr: Target learning rate at the end of warmup
        
    Note:
        Despite the name, self.last_epoch inherited from _LRScheduler 
        actually counts steps, not epochs. It starts at -1 and is 
        incremented by 1 every time scheduler.step() is called.
    """
    def __init__(self, optimizer, total_steps, final_lr):
        self.total_steps = total_steps
        self.final_lr = final_lr
        super().__init__(optimizer)  # last_epoch=-1 by default

    def get_lr(self):
        # self.last_epoch is actually the current step number (starts at 0)
        current_step = self.last_epoch
        # Calculate current step's learning rate
        progress = float(current_step) / self.total_steps
        # Clip progress to avoid lr going above final_lr
        progress = min(1.0, progress)
        
        return [self.final_lr * progress for _ in self.base_lrs]

def visualize_point_cloud_batch(batch_xyz, title="Batch Point Clouds"):
    """
    Visualize a batch of point clouds using Plotly.
    
    Args:
        batch_xyz (torch.Tensor): A tensor of shape (bs, l, 3) containing XYZ point clouds.
        title (str): Plot title.
    """
    assert batch_xyz.ndim == 3 and batch_xyz.shape[2] == 3, "Input must be of shape (bs, l, 3)"
    bs = batch_xyz.shape[0]

    # Determine subplot grid size
    rows = int(math.ceil(math.sqrt(bs)))
    cols = int(math.ceil(bs / rows))

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scatter3d'}]*cols for _ in range(rows)],
        subplot_titles=[f"Sample {i}" for i in range(bs)],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    for idx in range(bs):
        row = idx // cols + 1
        col = idx % cols + 1

        x, y, z = batch_xyz[idx, :, 0], batch_xyz[idx, :, 1], batch_xyz[idx, :, 2]
        scatter = go.Scatter3d(
            x=x.cpu().numpy(), y=y.cpu().numpy(), z=z.cpu().numpy(),
            mode='markers',
            marker=dict(
                size=3,
                color=z.cpu().numpy(),  # color based on z
                colorscale='Viridis',
                opacity=0.8
            )
        )

        fig.add_trace(scatter, row=row, col=col)

    fig.update_layout(
        height=300*rows, width=300*cols,
        title_text=title,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()


def compute_lddt(ground_truth_atoms, predicted_atoms, cutoff=30.0, thresholds=[1.0, 2.0, 4.0, 8.0]):
    """
    Computes the lDDT score between ground truth and predicted atoms.
    
    Parameters:
        ground_truth_atoms (np.array): Nx3 array of ground truth atom coordinates.
        predicted_atoms (np.array): Nx3 array of predicted atom coordinates.
        cutoff (float): Distance cutoff in Ångstroms to consider neighbors. Default is 30 Å.
        thresholds (list): List of thresholds in Ångstroms for the lDDT computation. Default is [0.5, 1.0, 2.0, 4.0].
    
    Returns:
        float: The lDDT score.
    """
    # Number of atoms
    num_atoms = ground_truth_atoms.shape[0]
    
    # Initialize array to store lDDT fractions for each threshold
    fractions = np.zeros(len(thresholds))
    
    for i in range(num_atoms):
        # Get the distances from atom i to all other atoms for both ground truth and predicted atoms
        gt_distances = np.linalg.norm(ground_truth_atoms[i] - ground_truth_atoms, axis=1)
        pred_distances = np.linalg.norm(predicted_atoms[i] - predicted_atoms, axis=1)
        
        # print(gt_distances)
        # print(pred_distances)
        # exit()
        # Apply the cutoff to consider only distances within the cutoff range
        mask = (gt_distances > 0) & (gt_distances < cutoff)
        
        # Calculate the absolute difference between ground truth and predicted distances
        distance_diff = np.abs(gt_distances[mask] - pred_distances[mask])

        # Filter out any NaN values from the distance difference calculation
        valid_mask = ~np.isnan(distance_diff)
        distance_diff = distance_diff[valid_mask]

        # Compute the fractions for each threshold
        for j, threshold in enumerate(thresholds):
            if len(distance_diff)>0:
                fractions[j] += np.mean(distance_diff < threshold)
    # print(fractions)
    # print(num_atoms)

    # Average the fractions over the number of atoms
    fractions /= num_atoms
    
    # The final lDDT score is the average of these fractions
    lddt_score = np.mean(fractions)
    
    return lddt_score

import os.path as path

class CSVLogger:
    def __init__(self,columns,file):
        self.columns=columns
        self.file=file
        if not self.check_header():
            self._write_header()


    def check_header(self):
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header


    def _write_header(self):
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

    def log(self,row):
        if len(row)!=len(self.columns):
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file,"a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)






def random_rotation_point_cloud_torch(point_cloud):
    """
    Apply a random 3D rotation to a Nx3 point cloud (PyTorch version).
    
    Args:
        point_cloud (torch.Tensor): Nx3 tensor of XYZ points.

    Returns:
        torch.Tensor: Rotated Nx3 point cloud.
    """
    # Generate a random 3x3 matrix
    A = torch.randn(3, 3).to(point_cloud.device)
    # QR decomposition to get an orthonormal matrix Q
    Q, R = torch.linalg.qr(A)
    # Ensure it's a proper rotation matrix (det = +1)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    rotated = point_cloud @ Q.T
    #add random rotation as well
    

    return rotated

def random_rotation_point_cloud_torch_batch(point_clouds):
    """
    Apply a random 3D rotation to a batch of point clouds (PyTorch version).
    
    Args:
        point_clouds (torch.Tensor): BxNx3 tensor of XYZ points.

    Returns:
        torch.Tensor: Rotated BxNx3 point clouds.
    """
    B, N, _ = point_clouds.shape
    device = point_clouds.device

    # Generate a batch of random orthonormal rotation matrices
    A = torch.randn(B, 3, 3, device=device)
    Q, R = torch.linalg.qr(A)

    # Ensure det(Q) = +1 for proper rotation
    det = torch.det(Q)
    Q[det < 0, :, 0] *= -1

    # Apply batched matrix multiplication
    rotated = torch.matmul(point_clouds, Q.transpose(1, 2))  # (B, N, 3) x (B, 3, 3)^T -> (B, N, 3)

    return rotated


def calculate_distance_matrix(X,Y,epsilon=1e-4):
    return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()


def dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    if d_clamp is not None:
        rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
    else:
        rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

    return rmsd.sqrt().mean()/Z


def dMSE(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    if d_clamp is not None:
        mse=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
    else:
        mse=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

    return mse.mean()


def local_dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=30):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=(~torch.isnan(gt_dm))*(gt_dm<d_clamp)
    mask[torch.eye(mask.shape[0]).bool()]=False



    rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon
    # rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).sqrt()/Z
    #rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])/Z
    #return rmsd.sqrt().mean()/Z
    return rmsd.mean()

def dRMAE(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])

    return rmsd.mean()/Z

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

    #mask 
    mask=~torch.isnan(target.sum(-1))

    input=input[mask]
    target=target[mask]
    
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
    return torch.abs(aligned_input-target).mean()/Z


def align_svd_rmsd(input, target):
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

    #mask 
    mask=~torch.isnan(target.sum(-1))

    input=input[mask]
    target=target[mask]
    
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
    return torch.square(aligned_input-target).mean().sqrt()

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


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, max_native: int = 40) -> float:
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
    os.system("mkdir pdbs")

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
        
        

        all_scores=[]
        for pred_cnt in range(1,6):
            tmp=[]
            for native_cnt in range(1,max_native+1):
                # Write solution PDB
                native_pdb=f'pdbs/{pdb_id}_native_{native_cnt}.pdb'
                predicted_pdb=f'pdbs/{pdb_id}_predicted_{pred_cnt}.pdb'

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
        #print(pdb_id)
        #print(all_scores)
        results_per_sub.append(all_scores)
        results.append(max(all_scores))
    
    #print(results)
    #return sum(results)/len(results), outputs
    return results, results_per_sub, outputs