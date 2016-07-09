import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import math
import scipy.io as sio

# print('\n', 'Begin script')

mat = sio.loadmat('ex3data1.mat')

X = np.array(mat['X'])

y = np.array(mat['y'])

# Not sure if we'll need this anywhere?
X_row_length = len(X[:, 0])
X_column_length = len(X[0, :])

# x = m x n
# theta = n x 1
# y = m x 1
m = len(y)



# x is X but with a column of 1's in the first column
# becoming a 5000 by 401
x = np.column_stack((ones(shape(X)[0]), X))
x_column_length = len(x[0, :])
n = x_column_length

# print(m)

theta = zeros(shape=(n,1))

z = dot(x, theta)

# print(z)

def sigmoid_function(z):
    hypothesis = 1.0 / (1.0 + (math.e)**(-z))
    #print(hypothesis)
    return hypothesis

# print(sigmoid_function(z))
# theta gets +1 because that'll be our bias weight ???
# initial_theta = zeros(shape=(m+1,1))


exlist = y
one_v_all = zeros(shape=(n,1))
for j in range(9):
	for i in range(m):
		if exlist[i] == j:
			one_v_all[i] = 1
		else:
			one_v_all[i] = 0
exlist = "exlist" + str(j)


