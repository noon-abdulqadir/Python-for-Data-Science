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

from random import choice
from numpy import array, dot, random
import numpy as np

# +
#MC Neuron for And

w = random.rand(2)
w[0] = 1
w[1] = 1

training_data = [
    (array([0,0]), 0),
    (array([0,1]), 0),
    (array([1,0]), 0),
    (array([1,1]), 1),
]
training_data
# -


