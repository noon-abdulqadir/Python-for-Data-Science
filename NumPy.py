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

import numpy as np

# +
# Single dimensional numpy array

n1 = np.array([10,20,30,40,50])
print(n1)
# -

type(n1)

# +
# Multi dimensional numpy array

n2 = np.array([[10,20,30],[30,20,10]])
print(n2)
# -

type(n2)

n1 = np.zeros((1,2))
print(n1)

n1 = np.zeros((6,6))
print(n1)

n1 = np.full((3,3),7)
print(n1)

n1 = np.arange(1,11)
print(n1)

n1 = np.arange(100,300,50)
print(n1)

n1 = np.random.randint(100,200,10)
print(n1)

n1 = np.array([[1,2,3,4],[5,6,7,8]])
n1

n1.shape

n1.shape = (4,2)
n1

a = np.array([10,20])
b = np.array([30,40])

np.sum([a,b])

np.sum([a,b],axis=0)

np.sum([a,b],axis=1)

n1 = np.array([[10,20,30]])
n2 = np.array([[30,40,50]])

np.vstack([n1,n2])

np.hstack([n1,n2])

np.column_stack([n1,n2])
