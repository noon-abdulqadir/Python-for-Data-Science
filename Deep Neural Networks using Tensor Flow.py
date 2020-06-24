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

# # Deep and Artifical Neural Networks (ANN)

# Artifical Neural networks have many hidden layers of neurons, that only get/receive data from the layers next to them. This makes it a DEEP neural network. More classification requires equal number of neurons whereas binary classification only needs 2 neurons. The hidden layers do not interact with the world outside. The more layers we have, the deeper the network becomes, the more difficult it becomes to train it. Input layer is our input data (columns in the df, called dummy neurons). We add bias neurons (also called dummy neurons) thanks to Rosenbalt. The bias goes to the neurons based on the weights of each neuron. These weights are modified to reduce the error.
#
# Examples of ANN: Google Inception and Microsoft ResNet

# +
import pandas as pd
import numpy as np
import tensorflow as tf

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print(tf.__version__)

# +
# White wine
white = pd.read_csv("winequality-white.csv",sep=";")
#Get files from the web: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

# Red wine
red = pd.read_csv("winequality-red.csv",sep=";")
#Get files from the web: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

# +
# Print info on white wine

print(white.info())

# Print info on red wine

print(red.info())
# -

white.describe()

red.describe()

pd.isnull(white).count()

pd.isnull(red).count()

# +
# Add 'type' column to 'white' with value 0
white['type'] = 0

# Add 'type' column to 'red' with value 1. Note: the underrepresented class usually gets the code 1
red['type'] = 1

# Append 'white' to 'red'
wines = white.append(red, ignore_index=True)
wines.tail()
# -

# The pair-plot (bi-variate visualization of rows and density) allows us to see whether a particular attribute/variable is a good candidate with which the data can be classified by the ANN. If there is significant overlap in density (number of data points per unit, i.e. length of the attribute), especially if the peaks of the diagonal histograms overlap, the algorithm can't use it to classify, i.e. the attribute/variables is a weak attribute for classification. The attributes should have the two classes be separated significantly for the null hypothesis to be rejected.
#
# Good discriminators below are:
# 1. Chloride
# 2. Total sulphur dioxide
#
# In ANN, the belief is that individually, the attributes may be weak, but taken together they become strong attributes.

import seaborn as sns
wines.head(50)
sns.pairplot(wines, diag_kind = "kde", hue = "type") #always start analysis of pairplots from the diagonals

# Whenever we are dealing with deep neural network based models, the first version will be rarely satisfactory. So we tweak the hyperparameters to make it perform the way we expect it to perform. Whenever we work with hyperparameters, its best to have at least 3 datasets:
# 1. Training
# 2. Validation
# 3. Testing
#
# In the model.fit section, we specify a validation split.
#
# Training and validation datasets are used to tweak the hyperparameters (e.g. k number of neighbours or the depth of a decision tree, which are subsumed within the model) whereas the test dataset is used in the last step to test the performance. This is done to prevent "data leaks" which can happen in different ways. One way is when we split the dataset into training and test sets, and use that testing dataset for hyperparameter tuning as well. We will then not have a new dataset for the model to predict that the model has not already experienced. Hyperparameter values in this case are already a function of the test data, much like a leaked exam paper. The model will thus not perform well in the real world. So we tweak hyperparameters on the validation dataset.
#
# Sometimes, we do not have enough data to split it into three datasets. Moreover, by spliting datasets (particularly small datasets), we are modifying the distribution, hence introducing bias errors. When the dataset is not big enough to split into three (without introducing bias errors), then we use k-fold cross-validation.

# +
# Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split

# Specify the data
X = wines.iloc[:,0:11]

# Specify the target labels and flatten the array

y = np.ravel(wines.type) # or np.ravel(wines['type'])
#y = wines.type

# Split the data up in the train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# To obtain a validation dataset

# Split the data up in the train and test sets
#X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

#X_train, X_train_val, y_train, y_train_val = train_test_split(X_train_val, y_train_val, test_size = 0.30, random_state = 42)
# -

y_test[0:10]

# +
# Import 'Standardcaler' from 'sklearn.preprocessing'
from sklearn.preprocessing import StandardScaler # Standardizes the data (z-scoring), instead of the raw numbers (X-bar * std)

# Define the scaler
scaler = StandardScaler().fit(X_train) # Do not apply the standard scaler on the output variables, only input

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# When you have a validation dataset

# Scale the train set
#X_train = scaler.transform(X_train)

# Scale the validation set
#X_train = scaler.transform(X_train_val)

# +
X_train.size

# When you have a validation dataset
#X_train_val.size

# +
# Using Tensorflow Keras instead of the original Keras # Keras comes bundled with tensorflow

from tensorflow.keras import Sequential # Sequential = simpler to learn, and functional = more versatile and flexible
from tensorflow.keras.layers import Dense # Layer, CNN (Convolutional neural network) also has convolution and pooling layers
from keras.layers.advanced_activations import ReLU # can be sigmoid, TanH, etc.
from tensorflow.keras.callbacks import EarlyStopping # helps determine the optimal number of epochs
# used as follows: callbacks=[EarlyStopping(monitor='val_loss', patience=3)] in model.fit

# Define the model architecture

# Initialize the constructor aka instantiating
model = Sequential()

# Add an input layer (hidden layer)
model.add(Dense(30, activation = "relu", input_shape = (11,)))
#30 is density (# of neurons) and is just random, we have to try out different numbers, but keep it at least equal to the number of attributes in the dataset
#11 is for the number of columns (independent attributes)
# We can add more layers

# Add one hidden layer
model.add(Dense(20, activation = "relu"))

# Add an outpuy layer
model.add(Dense(1, activation = "sigmoid")) # 1 layer because its a binary classification, sigmoid = logistic regression

# Add an input layer
#model.add(Dense(10, activation = "relu", input_shape = (11,)))

# Add an input layer
#model.add(Dense(20, activation = "relu", input_shape = (11,)))
# -
# To know whether you are overfitting or underfitting, you should run your model against training and validation datasets simultaneously. You can also write your own activation models.


# 1. Keep the epochs 20
# 2. Keep the batch size 5000
# 3. Check the accuracy (poor accuracy score, discuss)
# 4. Increase epochs to 40
# 5. Check the accuracy (high accuracy score, discuss)
# 6. Reduce batch size to 256
# 7. Reduce epochs to 20
# 8. Check accuracy score (90%+, discuss)
# 9. Add more hidden layers, what is the observation?
# 10. Replace 'relu' with 'sigmoid', what is the observation?

# +
model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
             metrics = ["accuracy"])

epochs = 40
batch_size = 50017
# -

# To estimate errors from predicted values we get from the model, we define error function as binary_crossentropy. To reduce this error, we use the optimizer adam (or sgd, rmsprop, etc.). In this step the model transforms from a blueprint to a concrete object (think of how re.compile becomes an object when applied). It also automatically adds bias at this step (by deafult its xavier methodology of initializing the weights).
#
# Epochs is one full read of the training dataset to estimate the weights based on the error, i.e. one round of forward and backward propagation to minimize the error. We cannot decide the number of epochs before hand, but there is method to prevent wastage of computational resource called 'early stopping'.
#
# To prevent outliers from skewing our data, do an unsupervised clustering to see these outliers. Also, hierarchical clustering.

history = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_split=0.3, verbose=True)
loss,accuracy = model.evaluate(X_test, y_test,verbose=False)

model.summary()

# +
print(history.history["accuracy"])
print(history.history["val_accuracy"])

ta = pd.DataFrame(history.history["accuracy"])
va = pd.DataFrame(history.history["val_accuracy"])

tva = pd.concat([ta,va], axis=1)

tva.boxplot()
# -

# **NOTE:** If the boxplots of the training and testing don't overlap, it's a clear indication of overfitting.

y_pred = np.round(model.predict(X_test))

y_pred[0:10]

# +
loss,acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: %.3f"%acc)
print("Test Loss: %.3f"%loss)

loss,acc = model.evaluate(X_train, y_train, verbose=0)
print("Train Accuracy: %.3f"%acc)
print("Train Loss: %.3f"%loss)

# +
from sklearn import metrics

df_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred, labels=[0,1]),
                         columns=['pred:White','pred:Red'],index=['true:White','true:Red'])
print(df_matrix)

# +
# %matplotlib inline
from matplotlib import pyplot
from numpy import where

pyplot.figure(figsize=(10,6))  
sns.heatmap(df_matrix, annot=True)

# +
# %matplotlib inline

# Plot loss during training
pyplot.subplot(211)
pyplot.title("Loss")
pyplot.plot(history.history["loss"], label = "Train")
pyplot.legend()
