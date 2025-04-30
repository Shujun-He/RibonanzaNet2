import pandas as pd
import matplotlib.pyplot as plt

# --- CASP15 ---
df1 = pd.read_csv('casp15_scores.csv')
df2 = pd.read_csv('../test10_rnet1/casp15_scores.csv')
df3 = pd.read_csv('../test10_nopretrain/casp15_scores.csv')

df_merged = df1[['target_id', 'TM-score']].rename(columns={'TM-score': 'Rnet2'}).merge(
    df2[['target_id', 'TM-score']].rename(columns={'TM-score': 'Rnet1'}), on='target_id', how='outer'
).merge(
    df3[['target_id', 'TM-score']].rename(columns={'TM-score': 'Rnet2_no_pretrain'}), on='target_id', how='outer'
).set_index('target_id').sort_index()

means_15 = df_merged.mean()
legend_labels_15 = [f"{col} (avg: {means_15[col]:.3f})" for col in df_merged.columns]

# --- CASP16 ---
df1 = pd.read_csv('casp16_scores.csv').iloc[3:].reset_index(drop=True)
df2 = pd.read_csv('../test10_rnet1/casp16_scores.csv').iloc[3:].reset_index(drop=True)
df3 = pd.read_csv('../test10_nopretrain/casp16_scores.csv').iloc[3:].reset_index(drop=True)

df_merged2 = df1[['target_id', 'TM-score']].rename(columns={'TM-score': 'Rnet2'}).merge(
    df2[['target_id', 'TM-score']].rename(columns={'TM-score': 'Rnet1'}), on='target_id', how='outer'
).merge(
    df3[['target_id', 'TM-score']].rename(columns={'TM-score': 'Rnet2_no_pretrain'}), on='target_id', how='outer'
).set_index('target_id').sort_index()

means_16 = df_merged2.mean()
legend_labels_16 = [f"{col} (avg: {means_16[col]:.3f})" for col in df_merged2.columns]

# --- Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True)

# CASP15 subplot
df_merged.plot(kind='bar', ax=axes[0])
axes[0].set_title('TM-score Comparison - CASP15')
axes[0].set_xlabel('Target ID')
axes[0].set_ylabel('TM-score')
axes[0].legend(legend_labels_15, title='Dataset')
axes[0].tick_params(axis='x', rotation=45)

# CASP16 subplot
df_merged2.plot(kind='bar', ax=axes[1])
axes[1].set_title('TM-score Comparison - CASP16')
axes[1].set_xlabel('Target ID')
axes[1].set_ylabel('TM-score')
axes[1].legend(legend_labels_16, title='Dataset')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('TM_score_comparison_casp15_casp16.png')
plt.show()
