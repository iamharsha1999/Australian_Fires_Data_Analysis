import pandas as pd
import os
import numpy as np

files = ['CSV_2.csv','CSV_3.csv']

df = pd.read_csv('Dataset/CSV_2.csv')

df['confidence'] = pd.to_numeric(df['confidence'])

for index, row in df.iterrows():
    if df.loc[index,'confidence'] <= 33.33:
        df.loc[index,'confidence'] = 'l'
    elif (df.loc[index,'confidence'] > 33.33) & (df.loc[index,'confidence'] <= 66.66):
        df.loc[index,'confidence'] = 'n'
    elif df.loc[index,'confidence'] > 66.66:
        df.loc[index,'confidence'] = 'h'

df.reset_index(drop = True)
df.to_csv('Dataset/CSV_2_modified.csv')
print('saved')
