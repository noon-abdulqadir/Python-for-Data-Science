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

# # Artifical Neural Networks (ANN)

# Artifical Neural networks have many hidden layers of neurons, that only get/receive data from the layers next to them. More classification requires equal number of neurons whereas binary classification only needs 2 neurons. The hidden layers do not interact with the world outside. The more layers we have, the deeper the network becomes, the more difficult it becomes to train it. Input layer is our input data (columns in the df, called dummy neurons). We add bias neurons (also called dummy neurons) thanks to Rosenbalt. The bias goes to the neurons based on the weights of each neuron. These weights are modified to reduce the error.
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

# Whenever we are dealing with deep neural network based models, the first version will be rarely satisfactory. So we tweak the hyperparameters to make it perform the way we expect it to perform. Whenever we work with hyperparameters, its best to have at least 3 datasets

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
# -

y_test[0:10]
