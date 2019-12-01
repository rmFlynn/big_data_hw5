import numpy as np


# 2 Stochastic gradient descent

n = 0.0001
e = 0.001
xpath = "./features.txt"
ypath = "./target.txt"

def get_data(xpath, ypath):
    x = np.genfromtxt(xpath, delimiter=",")
    y = np.genfromtxt(ypath)
    y = y.reshape(len(y),1)
    return np.concatenate((x,y), axis=1)

xy = get_data(xpath, ypath)
gx = lambda xy :  xy[list(range(len(xy)-1))]
gy = lambda xy :  xy[len(xy)-1]

gx(xy[1,:])
gy(xy[1,:])
