import pandas  as pd

df=pd.read_csv('casp16_scores.csv')

#print(df.loc[3:,'TM-score'])
print(df.loc[3:,'TM-score'].mean())

df=pd.read_csv('casp15_scores.csv')

#print(df.loc[3:,'TM-score'])
print(df['TM-score'].mean())

