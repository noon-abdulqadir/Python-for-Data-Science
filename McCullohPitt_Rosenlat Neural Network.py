# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# For more, go to:
# https://www.youtube.com/watch?time_continue=5392&v=JMxbypF825w

# ### Artifical Neurons

# Rosenblat's Perceptron is an artifical neuron, the forefather of Neural Networks. They failed because they couldn't work with with X OR gates (explained at the end) and Neural Networks are called "Universal Function Approximators".

from random import choice
from numpy import array, dot, random
import numpy as np

# +
#MC Neuron for And Gate

w = random.rand(2)
w[0] = 1
w[1] = 1
print(w)

training_data = [
    (array([0,0]), 0),
    (array([0,1]), 0),
    (array([1,0]), 0),
    (array([1,1]), 1),]

print(training_data)

step_function = lambda x: 0 if x < 2 else 1 #Step function with threashold of 2. Anything below is 0

print("-"* 60)

for x, _ in training_data:
    result = dot(x, w)

    print(f"{x[:2]}: {result} -> {step_function(result)}")

# +
#MC Neuron for Or Gate

w = random.rand(2)
w[0] = 1
w[1] = 1
print(w)

training_data = [
    (array([0,0]), 0),
    (array([0,1]), 1),
    (array([1,0]), 1),
    (array([1,1]), 1),]

print(training_data)

step_function = lambda x: 0 if x < 1 else 1 #Step function with threashold of 1. Anything below is 0

print("-"* 60)

for x, _ in training_data:
    result = dot(x, w)

    print(f"{x[:2]}: {result} -> {step_function(result)}")

w = random.rand(2)
w[0] = 1
w[1] = 1
print(w)

training_data = [
    (array([0,0]), 1),
    (array([0,1]), 1),
    (array([1,0]), 1),
    (array([1,1]), 0),]
print(training_data)

step_function = lambda x: 0 if x >= 2 else 1 #Step function with threashold of > 2 is 0.

print("-"* 60)

for x, _ in training_data:
    result = dot(x, w)

    print(f"{x[:2]}: {result} -> {step_function(result)}")
# -

# Rosenblat's Perceptron included a way to adjust the weights and find the appropriate combination 
# to overcome the need to modify threasholds for each gate separately, it used a bias term using
# neuron can be modified to implement multiple Boolean functions in one code

# +
# Rosenblat's Neuron

step_function = lambda x: 0 if x < 50 else 1 #Step function with threashold of 0.5. Anything below is 0.

training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 0),
    (array([1,0,1]), 0),
    (array([1,1,1]), 1),]

print(training_data)

w = random.rand(3)
b = .1        # initializing bias term
errors = []
eta = 0.1     # learning rate
n = 10000

for _ in range(n):
    x, expected = choice(training_data)

    #w = np.append(w,b)

    result = dot(w,x)
    error = expected - step_function(result)   # irrespective of what threashold we set, the algorithm
    errors.append(error)
    w += eta * error * x

print("-"* 60)    

for x, _ in training_data:
    result = dot(x, w)

    print(f"{x[:3]}: {result} -> {step_function(result)}")

print("Weights: ",w)
# -

# Rosenblat's Perceptron will never work for X OR Gate because it cannot separate non-linear classes (in 2-D).
#
# X OR gate:
# ```python
# training_data = [
#     (array([0,0,1]), 1),
#     (array([0,1,1]), 0),
#     (array([1,0,1]), 0),
#     (array([1,1,1]), 1),]```
#
# However many times the below code is run, it will never get the correct answer two times in a row.

# +
# Rosenblat's Neuron used on X OR Gate

step_function = lambda x: 0 if x < 50 else 1 #Step function with threashold of 0.5. Anything below is 0.

training_data = [
    (array([0,0,1]), 1),
    (array([0,1,1]), 0),
    (array([1,0,1]), 0),
    (array([1,1,1]), 1),]

print(training_data)

w = random.rand(3)
b = .1        # initializing bias term
errors = []
eta = 0.1     # learning rate
n = 10000

for _ in range(n):
    x, expected = choice(training_data)

    #w = np.append(w,b)

    result = dot(w,x)
    error = expected - step_function(result)   # irrespective of what threashold we set, the algorithm
    errors.append(error)
    w += eta * error * x

print("-"* 60)    

for x, _ in training_data:
    result = dot(x, w)

    print(f"{x[:3]}: {result} -> {step_function(result)}")

print("Weights: ",w)
# -

# # Artifical Neural Networks (ANN)

# Artifical Neural networks have many hidden layers of neurons, that only get/receive data from the layers next to them. More classification requires equal number of neurons whereas binary classification only needs 2 neurons. The hidden layers do not interact with the world outside. The more layers we have, the deeper the network becomes, the more difficult it becomes to train it. Input layer is our input data (columns in the df, called dummy neurons). We add bias neurons (also called dummy neurons) thanks to Rosenbalt. The bias goes to the neurons based on the weights of each neuron. These weights are modified to reduce the error.
#
# Examples of ANN: Google Inception and Microsoft ResNet

import tensorflow as tf


