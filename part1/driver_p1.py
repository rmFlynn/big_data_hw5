import pandas as pd
import numpy as np
from gd1 import batch_gd
from gd2 import stochastic_gd
from gd3 import minibatch_gd

cost_data = pd.DataFrame()

C = 100
n = 0.0000003
e = 0.25
xpath = "./features.txt"
ypath = "./target.txt"
costs = batch_gd(C, n, e, xpath, ypath)
tmp_data = pd.DataFrame(np.asmatrix(costs).T, columns=['costs'])
tmp_data['Algorithm'] = "Batch GD"
tmp_data['Epoch (k)'] = list(tmp_data.index)
cost_data = cost_data.append(tmp_data)

C = 100
n = 0.0001
e = 0.001
xpath = "./features.txt"
ypath = "./target.txt"
costs = stochastic_gd(C, n, e, xpath, ypath)
tmp_data = pd.DataFrame(np.asmatrix(costs).T, columns=['costs'])
tmp_data['Algorithm'] = "Stochastic GD"
tmp_data['Epoch (k)'] = list(tmp_data.index)
cost_data = cost_data.append(tmp_data)

C = 100
n = 0.00001
e = 0.01
batch_size = 20
xpath = "./features.txt"
ypath = "./target.txt"
costs = minibatch_gd(C, n, e, batch_size, xpath, ypath)
tmp_data = pd.DataFrame(np.asmatrix(costs).T, columns=['costs'])
tmp_data['Algorithm'] = "Mini Batch GD"
tmp_data['Epoch (k)'] = list(tmp_data.index)
cost_data = cost_data.append(tmp_data)

import seaborn as sns
import matplotlib.pyplot as plt

# Make a plot of the results, epoch vs losses
plot1 = sns.lineplot(x='Epoch (k)',
                     y='costs',
                     hue='Algorithm',
                     data=cost_data,
                     linewidth=1.5,
                     marker="o")
plot1.set(xlabel = "epoch (k)")
plot1.set(ylabel = "Cost Values")
plot1.set(title = "Cost vs Stochastic ")
plot1.figure.savefig('plot.png')
print("the figure has been generated!")
# Clear the stage
plt.clf()

