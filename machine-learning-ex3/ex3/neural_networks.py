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

m = len(y)


# x is X but with a column of 1's in the first column
# becoming a 5000 by 401
x = np.column_stack((ones(shape(X)[0]), X))

print(m)

# theta gets +1 because that'll be our bias weight ???
# initial_theta = zeros(shape=(m+1,1))
