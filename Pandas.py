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

import pandas as pd

# +
#Series
# -

s1 = pd.Series([10,20,30,40,50])
s1

s1 = pd.Series([10,20,30,40,50],index=['a','b','c','d','e'])
s1

d1 = {'k1':10,'k2':20,'k3':30}
s1 = pd.Series(d1)
s1

# +
#Dataframe
# -

student = {"Student_name":['Bob','Sam','Julia','Charles'],"Student_marks":[87,13,99,67],"Student_major":['Math','Communications','Psychology','Biology']}

df = pd.DataFrame(student)

df.head(2)

df.tail(2)

df.describe()

df.info()

df1 = df.iloc[0:3,0:2]
df1

df2 = df.iloc[:,[0,2]]
df2

df3 = df.loc[0:3,("Student_marks","Student_major")]
df3

x = df[df['Student_marks']>60]
x

y = df[(df['Student_marks']>60) & (df['Student_major'] == "Psychology") | (df['Student_major'] == "Biology")]
y
