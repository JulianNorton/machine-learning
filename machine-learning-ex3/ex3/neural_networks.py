import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import math
import scipy.io as sio

print('\n', 'Begin script')

mat = sio.loadmat('ex3data1.mat')

X = np.array(mat['X'])

y = np.array(mat['y'])

# Not sure if we'll need this anywhere?
X_row_length = len(X[:, 0])
X_column_length = len(X[0, :])

# x = m x n
# theta = n x 1
# y = m x 1
# 401 j's
# 5000 i's
m = len(y)

# x is X but with a column of 1's in the first column
# becoming a 5000 by 401
x = np.column_stack((ones(shape(X)[0]), X))
x_column_length = len(x[0, :])
n = x_column_length
theta = zeros(shape=(n,1))
z = dot(x, theta)
regularization_lambda = 1/m


def sigmoid_function(z):
    hypothesis = 1.0 / (1.0 + (math.e)**(-z))
    # print(hypothesis)
    return hypothesis

print(sigmoid_function(z))

#         i   j
# print(x[0, 0])

def cost_function(x,y,i,j):
    return (sigmoid_function(x[i]) - y[i]) * x[i, j]

def calc_partial_derivatives():
    calc_partial_derivatives = list()
    for j in range(n):
        if j == 0:
            partial_derivatives = list()
            for i in range(m):
                cost = cost_function(x,y,i,j)
                partial_derivatives.append(cost)
            partial_derivatives = (1/m) * np.sum(partial_derivatives)
            calc_partial_derivatives = np.asarray(partial_derivatives)
        else:
            partial_derivatives = list()
            for i in range(m):
                cost = cost_function(x,y,i,j) + (regularization_lambda * theta[j])
                partial_derivatives.append(cost)
            partial_derivatives = (1/m) * np.sum(partial_derivatives)
            calc_partial_derivatives = np.append(calc_partial_derivatives, partial_derivatives)

calc_partial_derivatives()










