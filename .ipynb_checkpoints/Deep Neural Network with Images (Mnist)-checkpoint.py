# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
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
batch_size = 512
# -

history = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_split=0.3, verbose=True)
loss,accuracy = model.evaluate(x_test, y_test,verbose=False)
