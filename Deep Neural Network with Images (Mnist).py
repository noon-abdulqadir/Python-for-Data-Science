# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3.9
#     language: python
#     name: python3
# ---

# # Deep Neural Network with Mnist

# +
import pandas as pd
import numpy as np
from matplotlib import pyplot
from numpy import where
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Tensorflow also has a graph approach (as opposed to keras) which is distributed training and some processes will run in parallel

# Using Tensorflow Keras with Mnist, a clean and de-noised dataset, or Fashion_Mnist datasets, a very noisey dataset
# Noise is residuals of a model, i.e. unexplained variance in a model

# %matplotlib inline
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
print(tf.__version__)
# -

(xtrain,ytrain),(xtest,ytest) = mnist.load_data()
# If we want a validation dataset, use train-test split on the xtrain and ytrain

xtrain.shape

pyplot.imshow(xtrain[1,:,:],cmap="gray")

ytrain[:50]

L = pd.DataFrame(ytrain)
L[0].value_counts()

# +
# Represent Training & Testing samples suitable for tensorflow backend, flattening the 28, 28 

x_train = xtrain.reshape(xtrain.shape[0],784).astype("float32")
x_test = xtest.reshape(xtest.shape[0],784).astype("float32")
# -

x_test.shape

x_train/=225
x_test/=225

# +
from tensorflow import keras

y_train = keras.utils.to_categorical(ytrain, 10)
y_test = keras.utils.to_categorical(ytest, 10)

# +
# Initialize the constructor

model = Sequential()

# Define model architecture

model.add(Dense(784,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(10,activation="softmax"))

# +
model.compile(loss='categorical_crossentropy',
             optimizer="adam", metrics=['accuracy'])

epochs = 20
batch_size = 20000
# -

history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                    validation_split=0.1, verbose=True)
loss,accuracy = model.evaluate(x_test, y_test,verbose=False)

model.summary()

# +
print(history.history["accuracy"])
print(history.history["val_accuracy"])

ta = pd.DataFrame(history.history["accuracy"])
va = pd.DataFrame(history.history["val_accuracy"])

tva = pd.concat([ta,va], axis=1)

tva.boxplot()

# +
loss,acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy: %.3f"%acc)
print("Test Loss: %.3f"%loss) # Loss is error

loss,acc = model.evaluate(x_train, y_train, verbose=0)
print("Train Accuracy: %.3f"%acc)
print("Train Loss: %.3f"%loss)
# -

y_predict = model.predict(x_test)

y_predict[0]

np.argmax(y_predict[0])

# +
y_pred = []

for val in y_predict:
    y_pred.append(np.argmax(val))

#print(y_pred)

# Convert 0 1 to 1 and 1 0 to 0
from sklearn import metrics

cm = metrics.confusion_matrix(ytest,y_pred)
print(cm)

# +
import seaborn as sns

pyplot.figure(figsize=(10,6))
sns.heatmap(cm, annot=True)

# +
# %matplotlib inline

# Plot loss during training
pyplot.subplot(211)
pyplot.title("Loss")
pyplot.plot(history.history["loss"], label = "Train")
pyplot.legend()
