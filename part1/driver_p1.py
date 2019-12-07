import numpy as np
# All this to draw pretty pictures
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Local imports
from gd1 import batch_gd
from gd2 import stochastic_gd
from gd3 import minibatch_gd

cost_data = pd.DataFrame()
time_data = []

C = 100
n = 0.0000003
e = 0.25
xpath = "./features.txt"
ypath = "./target.txt"
costs, timedt = batch_gd(C, n, e, xpath, ypath)
time_data.append(timedt)
tmp_data = pd.DataFrame(np.asmatrix(costs).T, columns=['costs'])
tmp_data['Algorithm'] = "Batch GD"
tmp_data['Epoch (k)'] = list(tmp_data.index)

cost_data = cost_data.append(tmp_data)

C = 100
n = 0.0001
e = 0.001
xpath = "./features.txt"
ypath = "./target.txt"
# It is not clear to me if cost is from the
# iteration or the complete data set.
# Uncomment which ever.
costs, timedt = stochastic_gd(C, n, e, xpath, ypath)
#costs, timedt = stochastic_gd(C, n, e, xpath, ypath, use_batch_cost = False)
time_data.append(timedt)
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
# It is not clear to me if cost is from the
# iteration or the complete data set.
# Uncomment which ever.
costs, timedt = minibatch_gd(C, n, e, batch_size, xpath, ypath)
#costs, timedt = minibatch_gd(C, n, e, batch_size, xpath, ypath, use_batch_cost = False)
time_data.append(timedt)
tmp_data = pd.DataFrame(np.asmatrix(costs).T, columns=['costs'])
tmp_data['Algorithm'] = "Mini Batch GD"
tmp_data['Epoch (k)'] = list(tmp_data.index)

cost_data = cost_data.append(tmp_data)

print("Total time for each gradient decent technique:\n")
for t in time_data:
    print(t)

print("\n\n")

# Make a plot of the results, epoch vs losses
plt.figure(figsize=(16, 6))
plot1 = sns.lineplot(x='Epoch (k)',
                     y='costs',
                     hue='Algorithm',
                     data=cost_data,
                     linewidth=1,
                     marker="o")
plot1.set(xlabel = "epoch (k)")
plot1.set(ylabel = "Cost Values")
plot1.set(ylim = (-5,80))
plot1.set(xlim = (-5,80))
#plot1.set(xscale = 'log')
#plot1.set(yscale = 'log')
plot1.set(title = "Cost vs Stochastic ")
plot1.figure.savefig('plot.png')
print("the figure has been generated!")
# Clear the stage
plt.clf()

