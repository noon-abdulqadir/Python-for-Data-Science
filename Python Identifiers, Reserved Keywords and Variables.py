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

# # Python Identifiers

# +
import keyword

print(keyword.kwlist)
# -

print(len(keyword.kwlist))

# +
rollnumber = 12345
name = "noon"
marks = 80.5

print(rollnumber,name,marks)
# -

a,b,c,d = 10,20,30,40
print(a,b,c,d)

rollnumber,name,marks = 12345,"noon",80.5
print(rollnumber,name,marks)

graph1 = graph2 = graph3 = "Data Science"
print(graph1,graph2,graph3)

print(id(graph1))
print(id(graph2))
print(id(graph3))

graph2 = "Python"
print(id(graph1))
print(id(graph2))
