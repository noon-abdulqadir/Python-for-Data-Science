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

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

x = np.arange(1,11)
x

y1 = x*2
y1

y2 = x*3
y2

# +
#Line plot

plt.plot(x,y1,color="green",linewidth=2,linestyle=":")
plt.plot(x,y2,color="blue",linewidth=2,linestyle=":")
plt.title("Line plot")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid(True)
plt.show()

# +
# Bar plot

students = {"Sam":30,"Bob":50,"Julia":70}

names = list(students.keys())
print(names)
marks = list(students.values())
print(marks)
# -

plt.bar(names,marks,color="yellow")
plt.title("Marks of students")
plt.xlabel("Names of students")
plt.ylabel("Marks")
plt.grid(True)
plt.show()

plt.barh(names,marks,color="yellow")
plt.title("Marks of students")
plt.xlabel("Names of students")
plt.ylabel("Marks")
plt.show()

# +
# Scatter plot

x = [4,7,3,9,1,6]
y1 = [9,1,2,6,4,9]
y2 = [4,7,8,1,2,6]

plt.scatter(x,y1)
plt.scatter(x,y2,color="red")
plt.title("Marks of students")
plt.xlabel("Names of students")
plt.ylabel("Marks")
plt.grid(True)

# +
# Historgram


