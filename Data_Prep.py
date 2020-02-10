import pandas as pd
import os
import numpy as np

filenames = []
csv_files = []
for file in os.listdir('Dataset'):
    filenames.append('Dataset/' + file)
    csv_files.append(pd.read_csv('Dataset/' + file))

j=1
for i in csv_files:
    if len(list(i))>14:
        i = i.iloc[:,:-2]
        print(len(list(i)))
        i.to_csv('Dataset/CSV_' + str(j) + '.csv')
    else:
        i = i.iloc[:,:-1]
        print(len(list(i)))
        i.to_csv('Dataset/CSV_' + str(j) + '.csv')
    print(list(i))
    print("File Saved")
    j+=1
