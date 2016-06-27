import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import math
import scipy.io as sio

print('\n', 'Begin script')

mat = sio.loadmat('ex3data1.mat')


def sigmoid_function(z):
    hypothesis = 1.0 / (1.0 + (math.e)**(-z))
    # print(hypothesis)
    return hypothesis

hypothesis = sigmoid_function(z)

# cost function
def logistic_cost_function():
    J = 0
    for i in range(m):
        J = J + (-y[i] * np.log1p(hypothesis[i]) - (1.0 - y[i]) * np.log1p(1.0 - hypothesis[i]))
    print(J)
    

X = mat['X']
y = mat['y']

print(X, 'X from mat X')
print(y, 'Y from mat Y')

