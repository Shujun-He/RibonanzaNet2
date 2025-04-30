from glob import glob
import os

pdbs=glob("pdbs/*_native_1.pdb")+glob("pdbs/*_predicted*.pdb")

os.system("mkdir recon_pdbs")

for pdb in pdbs:
    target_path=pdb.replace("pdbs/","recon_pdbs/")
    os.system(f"../Arena {pdb} {target_path}")


