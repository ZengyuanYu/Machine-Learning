import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def real_func(x):
    return np.sin(2*np.pi*x)


def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


regularize = 0.0001
def residual_func_regularize(p, x, y):
    ret = fit_func(p, x) - y 
    ret = np.append(ret, np.sqrt(0.5*regularize*np.square(p)))#L2范数
    return ret


def residual_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 10000)
y_ = real_func(x)
y = [np.random.normal(0, 0.1)+y1 for y1 in y_]

def fitting(M=0):

    p_init = np.random.rand(M+1)
    p_lsq = leastsq(residual_func, p_init, args=(x, y))
    print("Fitting params:", p_lsq[0])
     # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq

def fitting_regulariz(M=0):
    
    p_init = np.random.rand(M+1)
    p_lsq = leastsq(residual_func_regularize, p_init, args=(x, y))
    print("Fitting params:", p_lsq[0])
     # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq
print(fitting(M=9))
print(fitting_regulariz(M=9))

