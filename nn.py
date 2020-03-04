from keras.layers import Dense, Activation, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from keras.utils import get_custom_objects
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy  as np
import matplotlib.pyplot as plt

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

# ## Plot a correlation matrix
# corr = df.corr()
# mask = np.triu(np.ones_like(corr, dtype=np.bool))
# sns.heatmap(corr, mask=mask, vmax=.3, center=0,square=True, linewidths=0.5, cbar_kws={"shrink": .5})
# plt.show()

y = df.confidence.values
x = df.drop('confidence', axis = 1).reset_index(drop = True)
y = to_categorical(y)

x_train,x_val, y_train, y_val = train_test_split(x,y, test_size=0.3)

## Build and compile the model
model = build_model((x.shape[1],))
# sgd = SGD(lr = 0.0001, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adagrad', metrics = ['accuracy'])
model.summary()

# Checkpoints and Learning Rate Reducer
file_path = 'NN_weights/#108/weights-{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkpoints = ModelCheckpoint(file_path, verbose= 1, monitor= 'val_accuracy', save_best_only= True, mode = 'auto')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=7, min_lr=0.000001)
callbacks = [checkpoints, reduce_lr]

# Train the model
history = model.fit(x_train, y_train, epochs = 100, batch_size=32, validation_data = (x_train,y_train), callbacks = callbacks)

# Visualization
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('NN_weights/#108/Fig_1.png')

plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('NN_weights/#108/Fig_2.png')
