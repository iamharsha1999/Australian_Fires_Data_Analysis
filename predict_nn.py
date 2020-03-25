from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import keras.backend as K
from keras.utils import get_custom_objects, to_categorical
from keras.layers import Activation, Dense, BatchNormalization
from keras.models import Sequential
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return K.sigmoid(x) * x

get_custom_objects().update({'swish': Swish(swish)})


## Define the model architecture
def build_model(input_shape):
     model = Sequential()

     model.add(Dense(32, input_shape=input_shape))
     model.add(Activation('swish'))
     model.add(BatchNormalization())

     model.add(Dense(16))
     model.add(Activation('swish'))
     model.add(BatchNormalization())

     model.add(Dense(8))
     model.add(Activation('swish'))
     model.add(BatchNormalization())

     model.add(Dense(3))
     model.add(Activation('softmax'))

     return model
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
df = df.iloc[:,1:]

y = df.confidence.values
x = df.drop('confidence', axis = 1).reset_index(drop = True)
# y = to_categorical(y)

x_train,x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)
y_valv = y_val
y_val = to_categorical(y_val)
y_train = to_categorical(y_train)

## Build and compile the model
model = build_model((x.shape[1],))
model.load_weights('NN_weights/#106/weights-25-0.93.hdf5')
predictions = model.predict(x_val, verbose = 1)
predictions = np.argmax(predictions,axis = 1)
print(predictions)
print("Y_Val", y_val)

# Calculate Accuracy and other metrics
print("Classification Metrics:")
accuracyv = accuracy_score(predictions, y_valv)
print('Accuracy: %f' % accuracyv)
precision = precision_score(predictions, y_valv, average = 'macro')
print('Precision: %f' % precision)
recall = recall_score(predictions, y_valv, average = 'macro')
print('Recall: %f' % recall)
f1 = f1_score(predictions, y_valv, average = 'macro')
print('F1 score: %f' % f1)

labels = ['h','l','n']
cm = confusion_matrix(y_valv, predictions)
sum = cm.sum()
cm = cm * 100 / (sum)
df_cm = pd.DataFrame(cm, index = labels,columns = labels)
fig = plt.figure()
sns.heatmap(df_cm, annot=True)
fig.tight_layout()
plt.savefig('nn.png')
