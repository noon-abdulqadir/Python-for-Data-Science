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

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

student = {"Student_name":['Bob','Sam','Julia','Charles','Noon'],"Student_marks":[87,13,99,67,99],"Student_major":['Math','Communications','Psychology','Biology','Communications']}
df = pd.DataFrame(student)
df.head(2)

plt.hist(df['Student_marks'],bins=20)
plt.grid(True)

# +
# Box plot

df.boxplot(column='Student_marks',by='Student_name')
# -

sns.boxplot(x=df["Student_name"],y=df["Student_marks"])

# +
# Pie Chart

fruits = ["apple","mango","orange","banana"]
cost = [76,45,90,85]

plt.figure(figsize=(6,6))
plt.pie(cost,labels=fruits,autopct="%0.2f",shadow=True)
plt.show()
