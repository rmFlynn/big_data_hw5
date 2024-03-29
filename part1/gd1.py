import numpy as np
import time

# 1 Batch gradient descent

def get_data(xpath, ypath):
    """read in the data and make one data matrix"""
    x = np.genfromtxt(xpath, delimiter=",")
    y = np.genfromtxt(ypath)
    y = y.reshape(len(y),1)
    return np.concatenate((x,y), axis=1)

# functions to extract x or y from the one data matrix
gx = lambda xy :  xy[list(range(len(xy)-1))]
gy = lambda xy :  xy[len(xy)-1]

def calfunc(w, b, xy, C):
    """Calculate the svm function for cost"""
    ifmo0 = lambda x : x if x > 0 else 0
    cond = lambda xi, yi : ifmo0(1-yi*(np.dot(w,xi) + len(w)*b))
    cond_sum = lambda dat : cond(gx(dat), gy(dat))
    # Use the lambdas to compute the functions
    result = 0
    result += (1/2)*np.dot(w,w)
    result += C*sum(np.apply_along_axis(cond_sum, 1, xy))
    return result

def caldeltawj(j, w, b, xy, C):
    """calculate the value of the partial derivative of f with respect to wj"""
    ifmo1 = lambda a, b : 0 if a >= 1 else b
    cond = lambda xi, yi, j : ifmo1(yi*(np.dot(w,xi) + b), -1*(yi*xi[j]))
    cond_sum = lambda dat : cond(gx(dat), gy(dat), j)
    result = 0
    result += w[j]
    result += C*sum(np.apply_along_axis(cond_sum, 1, xy))
    return result


def caldeltab(w, b, xy, C):
    """calculate the derivative of F with respect to b"""
    ifmo1 = lambda a, b : 0 if a >= 1 else b
    cond = lambda xi, yi : ifmo1(yi*(np.dot(w,xi) + b), -1*(yi))
    cond_sum = lambda dat : cond(gx(dat), gy(dat))
    result = 0
    result += C*sum(np.apply_along_axis(cond_sum, 1, xy))
    return result

def rep_convo_time(s,e):
    """Convert time to human readable format"""
    timet = e - s
    minutes = int(timet // 60 % 60)
    seconds = timet % 60
    return "Convergance took {0} minutes and {1:.2f} seconds".format(minutes,seconds)

def calcost(func, func_1):
    """given two svm calculations calculate the cost"""
    result = 0
    result += np.abs(func_1 - func)
    result *= 100
    result /= func_1
    return  result


def batch_gd(C = 100, n = 0.0000003, e = 0.25, xpath = "./features.txt", ypath = "./target.txt"):
    """Perform gradient decent"""
    # Init computed values
    xy = get_data(xpath, ypath)
    b = 0
    w = np.zeros(len(gx(xy[0])-1))
    np.random.seed(42)
    startt = time.time()
    # start with infinite cost
    cost = float('inf')
    cost_hist = []
    # loop while conditional
    while(e < cost):
        # shuffle the data
        np.random.shuffle(xy)
        func = calfunc(w, b, xy, C)
        # adjust weights
        for j in range(len(w)):
            w[j] -= n*caldeltawj(j, w, b, xy, C)
        # adjust the intercept
        b -= n*caldeltab(w, b, xy, C)
        func_1 = calfunc(w, b, xy, C)
        # Calculate the cost
        cost = calcost(func, func_1)
        print("Iteration: {}, cost: {}".format(len(cost_hist), cost))
        cost_hist.append(cost)
    # Recode time and return values for plotting
    stopt = time.time()
    timemsg = rep_convo_time(startt,stopt)
    return cost_hist, timemsg


