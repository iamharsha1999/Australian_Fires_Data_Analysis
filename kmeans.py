from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le_s = LabelEncoder()
le_i = LabelEncoder()
le_c = LabelEncoder()
le_acd = LabelEncoder()


df = pd.read_csv('Dataset/Standard_CSV.csv')
df['satellite'] = le_s.fit_transform(df['satellite'])
df['instrument'] = le_i.fit_transform(df['instrument'])
df['confidence'] = le_c.fit_transform(df['confidence'])
df[['acq_date']] = df[['acq_date']].astype(str)
df['acq_date'] = le_acd.fit_transform(df['acq_date'])

y = df.confidence.values
x = df.drop('confidence', axis = 1).reset_index(drop = True)

km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, verbose = 0)

x_train,x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)

km.fit(x_train)
predictions = km.predict(x_val)
print(predictions)
