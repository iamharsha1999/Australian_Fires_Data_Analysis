from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, verbose = 1)

x_train,x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)

km.fit(x_train)
predictions = km.predict(x_val)

# Calculate Accuracy and other metrics
print("Classification Metrics:")
accuracyv = accuracy_score(predictions, y_val)
print('Accuracy: %f' % accuracyv)
precision = precision_score(predictions, y_val, average = 'macro')
print('Precision: %f' % precision)
recall = recall_score(predictions, y_val, average = 'macro')
print('Recall: %f' % recall)
f1 = f1_score(predictions, y_val, average = 'macro')
print('F1 score: %f' % f1)

# Plot Confusion Matrix
labels = ['h','l','n']
cm = confusion_matrix(y_val, predictions)
df_cm = pd.DataFrame(cm, index = labels,columns = labels)
fig = plt.figure()
sns.heatmap(df_cm, annot=True)
fig.tight_layout()
plt.savefig('kmeans.png')
