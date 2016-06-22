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

theta = zeros(shape=(m,3))

z = x * theta
# print(z)
def sigmoid_function(z):
    hypothesis = 1.0 / (1.0 + (math.e)**(-z))
    # print(hypothesis)
    return hypothesis

hypothesis = sigmoid_function(z)


test_list = list()

J = 0
for i in range(m):
    J = J + (-y[i] * np.log1p(hypothesis[i]) - (1.0 - y[i]) * np.log1p(1.0 - hypothesis[i]))

print(J)

# def cost_function(hypothesis, m, y):
    # J = (1.0/m) * np.sum(-y*np.log1p(hypothesis) - (1.0-y) * np.log1p(1.0 - hypothesis))
    # J = '0'
    # print(J)

# cost_function(hypothesis, m, y)
