import h5py
import numpy as np
import os

os.system('mkdir data')

data_dict = h5py.File('../../Ribonanza2A_Genscript.v0.1.0.hdf5', 'r')
SNR=data_dict['signal_to_noise'][:]

# new_data_dict['r_norm']=data_dict['r_norm'][:]
# new_data_dict['sequences']=data_dict['sequences'][:]
# new_data_dict['signal_to_noise']=data_dict['signal_to_noise'][:]

# #get top 1% indices as val
# high_snr_indices = np.where(SNR.min(1)>=1)[0]
# sorted_indices=SNR[high_snr_indices].mean(1).argsort()[::-1]
# top_snr_n=int(len(sorted_indices)*0.01) #use top 1% as val data
# val_indices=high_snr_indices[sorted_indices][:top_snr_n]

# usable_snr_indices = np.where(SNR.max(1)>=1)[0]

# high_quality_indices = np.where(SNR.min(1)>=1)[0]

# train_indices = usable_snr_indices[~np.isin(usable_snr_indices, val_indices)]#[:20000]
# train_indices.shape

# new_data_dict={}
# new_data_dict['r_norm']=data_dict['r_norm'][:]
# new_data_dict['sequences']=data_dict['sequences'][:]
# new_data_dict['signal_to_noise']=data_dict['signal_to_noise'][:]

#save as memmap  r_norm sequences signal_to_noise



# data_dict=new_data_dict
