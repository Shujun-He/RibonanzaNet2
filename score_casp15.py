#score val
import pandas as pd
import pandas.api.types
import os
import re
import numpy as np

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
            for native_cnt in range(1,41):
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
        print(pdb_id)
        print(all_scores)
        results_per_sub.append(all_scores)
        results.append(max(all_scores))
    
    print(results)
    #return sum(results)/len(results), outputs
    return results, results_per_sub, outputs

solution=pd.read_csv("../input/validation_labels.csv")
submission=pd.read_csv("submission.csv")

scores,results_per_sub,outputs=score(solution,submission,'ID')
print(np.mean(scores))

# Save the results to a CSV file

df = pd.DataFrame()
df['target_id'] = solution['pdb_id'].unique()
df['TM-score'] = scores
df[[f'TM-score-{i}' for i in range(1, 6)]] = pd.DataFrame(results_per_sub)

df.to_csv("casp15_scores.csv", index=False)