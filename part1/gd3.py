import numpy as np
import time


# 3 Mini batch gradient descent

def get_data(xpath, ypath):
    """read in the data and make one data matrix"""
    x = np.genfromtxt(xpath, delimiter=",")
    y = np.genfromtxt(ypath)
    y = y.reshape(len(y),1)
    return np.concatenate((x,y), axis=1)

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

def calcostcostkd(func, func_1, ck_1):
    """calkulat the diferace in cost one to two"""
    ck = 0
    ck += np.abs(func_1 - func)
    ck *= 100
    ck /= func_1
    ckd = 0.5*(ck) + 0.5*(ck_1)
    return  ck, ckd

def minibatch_gd(C = 100, n = 0.00001, e = 0.01, batch_size = 20, xpath = "./features.txt", ypath = "./target.txt", use_batch_cost = True):
    """Perform gradient decent"""
    # Init computed values
    xy = get_data(xpath, ypath)
    b = 0
    w = np.zeros(len(gx(xy[0])-1))
    np.random.seed(42)
    startt = time.time()
    # start with infinite cost
    cost = 0
    costkd = float('inf')
    cost_hist = []
    # loop while conditional
    costkd_hist = []
    while(e < costkd):
        # shuffle the data
        np.random.shuffle(xy)
        s0 = 0
        # iterat over values
        for s1 in range(batch_size, xy.shape[0], batch_size):
            batch = list(range(s0,s1))
            xyb = xy[batch]
            #print(xyb.shape)
            s0 = s1
            # It is not clear to me if cost is from the
            # iteration or the complete data set.
            # Uncomment which ever.
            if use_batch_cost:
                func = calfunc(w, b, xyb, C)
            else:
                func = calfunc(w, b, xy, C)
            for j in range(len(w)):
                w[j] -= n*caldeltawj(j, w, b, xyb, C)
            # adjust the intercept
            b -= n*caldeltab(w, b, xyb, C)
            if use_batch_cost:
                func_1 = calfunc(w, b, xyb, C)
            else:
                func_1 = calfunc(w, b, xy, C)
            # Calculate the cost
            cost, costkd = calcostcostkd(func, func_1, cost)
            print(cost)
            cost_hist.append(cost)
            costkd_hist.append(costkd)
            # break if condition met
            if e >= costkd:
                break;
    stopt = time.time()
    timemsg = rep_convo_time(startt,stopt)
    return cost_hist, timemsg


#minibatch_gd(C = 100, n = 0.00001, e = 0.01, batch_size = 20, xpath = "./features.txt", ypath = "./target.txt")

