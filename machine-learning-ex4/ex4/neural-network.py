import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import math
import scipy.io as sio

print('\n', 'Begin script')

hand_written_digits = sio.loadmat('ex4data1.mat')
X = np.array(hand_written_digits['X'])
y = np.array(hand_written_digits['y'])


pretrained_weights = sio.loadmat('ex4weights.mat')

theta1 = np.array(pretrained_weights['Theta1'])
theta2 = np.array(pretrained_weights['Theta2'])

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
alpha = .03


def sigmoid_function(z):
    hypothesis = 1.0 / (1.0 + (math.e)**(-z))
    # print(hypothesis)
    return hypothesis

#         i   j
# print(x[0, 0])

# def cost_function(x,y,i,j):
    # return (sigmoid_function(x[i]) - y[i]) * x[i, j]



print('\n', 'End script')
