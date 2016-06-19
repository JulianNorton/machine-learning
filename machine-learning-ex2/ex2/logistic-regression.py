import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import math

print('\n', 'logistic regression')

# Test scores
data_1 = np.loadtxt('ex2data1.txt',delimiter=',')
plt.scatter(data_1[:,0],data_1[:,1])
plt.xlabel('Exam #1 score')
plt.ylabel('Exam #2 score')

# plt.show()

# Admitted or not plot
data_2 = np.loadtxt('ex2data2.txt',delimiter=',')
plt.scatter(data_2[:,0],data_2[:,1])
# plt.show()


# Defining variables
x = data_1[:,[0,1]]
m = len(x)
y = data_1[:,2]

x_ones = ones(shape=(m, 3))
x_ones[:, [1,2]] = x
x = x_ones

# already transposed
theta = zeros(shape=(m,3))

z = theta * x

def sigmoid_function(z):
    hypothesis = 1 / (1 + (math.e)**(-z))
    print hypothesis
    return hypothesis

hypothesis = sigmoid_function(z)

# g = ((1.0/m)* np.sum(-y*math.log1p(hypothesis)) - (1-y)*log1p(1 - hypothesis))

# def cost_function(theta, x, y):
#     prediction_y = dot(x, theta)
#     J = (1.0 / (2*m)) * dot(transpose(prediction_y - y) , (prediction_y - y))
#     print(J, 'cost')
#     return J