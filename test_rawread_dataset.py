from Dataset import *
import polars as pl


input_dir = '/lustre/fs0/scratch/shujun/BothLanes_RPT'

raw_data = np.memmap(f'{input_dir}/RTB000_Marathon_Bicine_3pct_DMS.reads.txt.memmap', dtype=np.uint8, mode='r', shape=(606861064, 256))
csv=pl.read_csv(f'{input_dir}/RTB000_Marathon_Bicine_3pct_DMS.index.csv')

csv=csv.filter(csv["num_reads"]>128)

rawread_indices=[[i,j] for i,j in zip(csv['read_start'].to_numpy(),csv['read_end'].to_numpy())]
sequences=csv['sequence'].to_list()

dataset=RawReadRNADataset(np.arange(len(sequences)),raw_data,sequences,rawread_indices)

d=dataset[0]

for i in range(100):
    d=dataset[i]
    # print(d['sequence'].shape)
    # print(d['rawreads'].shape)
    # print(d['mask'].shape)

    if len(d["rawreads"])>256:
        print('rawreads length > 256')
        print(i)
    # print(d['sequence'])
    # print(d['rawreads'])
    # print(d['mask'])
    # print('')

    # if i==0:
    #     break