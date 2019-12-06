import numpy as np
import time


# 2 Stochastic gradient descent


def get_data(xpath, ypath):
    x = np.genfromtxt(xpath, delimiter=",")
    y = np.genfromtxt(ypath)
    y = y.reshape(len(y),1)
    return np.concatenate((x,y), axis=1)

gx = lambda xy :  xy[list(range(len(xy)-1))]
gy = lambda xy :  xy[len(xy)-1]


def calfunc(w, b, xy, C):
    ifmo0 = lambda x : x if x > 0 else 0
    cond = lambda xi, yi : ifmo0(1-yi*(np.dot(w,xi) + len(w)*b))
    cond_sum = lambda dat : cond(gx(dat), gy(dat))
    result = 0
    result += (1/2)*np.dot(w,w)
    result += C*sum(np.apply_along_axis(cond_sum, 1, xy))
    return result

def calfunc_sto(w, b, xy, C):
    ifmo0 = lambda x : x if x > 0 else 0
    cond = lambda xi, yi : ifmo0(1-yi*(np.dot(w,xi) + len(w)*b))
    cond_sum = lambda dat : cond(gx(dat), gy(dat))
    result = 0
    result += (1/2)*np.dot(w,w)
    result += C*cond_sum(xy)
    return result

def caldeltawj_sto(j, w, b, xyi, C):
    ifmo1 = lambda a, b : 0 if a >= 1 else b
    cond = lambda xi, yi, j : ifmo1(yi*(np.dot(w,xi) + b), -1*(yi*xi[j]))
    cond_sum = lambda dat : cond(gx(dat), gy(dat), j)
    result = 0
    result += w[j]
    result += C*(cond_sum(xyi))
    return result


def caldeltab_sto(w, b, xyi, C):
    ifmo1 = lambda a, b : 0 if a >= 1 else b
    cond = lambda xi, yi : ifmo1(yi*(np.dot(w,xi) + b), -1*(yi))
    cond_sum = lambda dat : cond(gx(dat), gy(dat))
    result = 0
    result += C*(cond_sum(xyi))
    return result

def rep_convo_time(s,e):
    timet = e - s
    minutes = int(timet // 60 % 60)
    seconds = timet % 60
    print("Convergance took {0} minutes and {1:.2f} seconds".format(minutes,seconds))


def calcost(func, func_1):
    result = 0
    result += np.abs(func_1 - func)
    result *= 100
    result /= func_1
    return  result


def calcostcostkd(func, func_1, ck_1):
    ck = 0
    ck += np.abs(func_1 - func)
    ck *= 100
    ck /= func_1
    ckd = 0.5*(ck) + 0.5*(ck_1)
    return  ck, ckd




def stochastic_gd(C = 100, n = 0.0001, e = 0.001, xpath = "./features.txt", ypath = "./target.txt", use_batch_cost = False):
    xy = get_data(xpath, ypath)
    b = 0
    w = np.zeros(len(gx(xy[0])-1))
    np.random.seed(124)
    startt = time.time()
    cost = 0
    costkd = float('inf')
    cost_hist = []
    costkd_hist = []
    while(e < costkd):
        np.random.shuffle(xy)
        for i in range(xy.shape[0]):
            xyi = xy[i]
            if use_batch_cost:
                func = calfunc_sto(w, b, xyi, C)
            else:
                func = calfunc(w, b, xy, C)
            for j in range(len(w)):
                w[j] -= n*caldeltawj_sto(j, w, b, xyi, C)
            b -= n*caldeltab_sto(w, b, xyi, C)
            if use_batch_cost:
                func_1 = calfunc_sto(w, b, xyi, C)
            else:
                func_1 = calfunc(w, b, xy, C)
            cost, costkd = calcostcostkd(func, func_1, cost)
            print(cost)
            cost_hist.append(cost)
            costkd_hist.append(costkd)
            if e >= costkd:
                break;
    stopt = time.time()
    rep_convo_time(startt,stopt)
    return cost_hist


#stochastic_gd(C = 100, n = 0.0001, e = 0.001, xpath = "./features.txt", ypath = "./target.txt")
