import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import math
import scipy.io as sio

print('\n', 'Begin script')

mat = sio.loadmat('ex3data1.mat')

print(mat)

print(mat['X'])
print(mat['y'])

X = mat['X']
y = mat['y']

print(X, 'X from mat X')
print(y, 'Y from mat Y')