from cProfile import label
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os




#Show running mean y-log plot
# colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
customPalette = sns.set_palette(sns.color_palette(colors))


# sigmoid function
# x = np.linspace(-10, 10, 1000)
# y = 1 / (1 + np.exp(-x))


# hyperbolic tangent
# x = np.linspace(-10, 10, 1000)
# y = ( 2 / (1 + np.exp(-2*x) ) ) -1


# Relu
# x = np.linspace(-10, 10, 1000)
# y = np.maximum(0, x)


# Exp growth
x = np.linspace(0, 8, 9)
y = pow(8, x)


# plot
plt.figure(figsize=(10, 7.5))
plt.plot(x, y, 'o-')
# plt.legend()

for x, y in zip(x, y):
    label = "{:.0f}".format(y)
    if y < 250000:
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    elif y > 250000 and y < 300000:
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(-5, 10), ha='center')
    else:
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(-8, 0), ha='right')

plt.title('Exponential Growth of Joint Action Space')
plt.xlabel('Number of Intersections')
plt.ylabel('Number of Actions')

plt.show()