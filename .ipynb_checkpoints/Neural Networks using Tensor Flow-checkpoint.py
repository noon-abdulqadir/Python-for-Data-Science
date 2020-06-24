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


