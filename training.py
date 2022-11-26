#import libraries

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Conv1D,Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
import mlflow.tensorflow
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy


#setup auto-log with mlflow
mlflow.tensorflow.autolog()

#setup hyperparameters

epochs=10
batch_size=32
learning_rate=0.001

#setup data
X,y = make_blobs(n_samples=50000, n_features=10, centers=10, random_state=25)
print(X.shape,y.shape)
plt.scatter(X[:,0], X[:, 1], c=y)
#plt.show()

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.33, random_state=25)
from tensorflow.keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

#model building
#create model
model = Sequential()
#add model layers
model.add(Dense(128,activation='relu', input_shape=(10,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(10, activation='softmax'))

print(model.summary())
#compile model using accuracy to measure model performance
model.compile(
    optimizer=Adam(learning_rate),
    loss=CategoricalCrossentropy(from_logits=False),
    metrics=[CategoricalAccuracy()],
)

#train the model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,batch_size=batch_size,verbose=1)

#evaluate model
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)


train_loss=history.history['loss'][-1]
train_acc=history.history['categorical_accuracy'][-1]
val_loss=history.history['val_loss'][-1]
val_acc=history.history['categorical_accuracy'][-1]

print("train_loss: ", train_loss)
print("train_accuracy: ", train_acc)
print("val_loss: ", val_loss)
print("val_accuracy: ", val_acc)

#experiments
#epochs=10,batch_size=16
#epochs=20,batch_size=16
#epochs=10,batch_size=32
#epochs=10,batch_size=32
import tensorflow as tf 

tf.keras.models.save_model(model, "./model")

run_name="Run-4"

with mlflow.start_run(run_name=run_name):
  mlflow.log_param("batch_size", batch_size)
  mlflow.log_param("epochs", epochs)
  mlflow.log_metric("train_loss", train_loss)
  mlflow.log_metric("train_accuracy", train_acc)
  mlflow.log_metric("val_loss", val_loss)
  mlflow.log_metric("val_accuracy", val_acc)
  mlflow.log_artifacts("./model")


 



