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
theta = zeros(shape=(3,1))
z = dot(x, theta)

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
# logistic_cost_function()

# def logistic_gradient_descent():

# print(x)
# row = 50
# column = 0
# j = column
# print(x[:, column])
j = len(x[0, :])

# for i in range(j):
#     print((i, j))

print(hypothesis[5])
# print(y[5])

sigma_x0, sigma_x1, sigma_x2 = list(), list(), list()

for i in range(m):
    sigma_x0.append(((hypothesis[i] - y[i]) * 1) * x[i, 0])
sigma_x0 = (1/m) * sum(sigma_x0)

for i in range(m):
    sigma_x1.append(((hypothesis[i] - y[i]) * 1) * x[i, 1])
sigma_x1 = (1/m) * sum(sigma_x1)

for i in range(m):
    sigma_x2.append(((hypothesis[i] - y[i]) * 1) * x[i, 2])
sigma_x2 = (1/m) * sum(sigma_x2)

print(sigma_x0, 'sigma_x0 \n', sigma_x1, 'sigma_x1 \n', sigma_x2, 'sigma_x2 \n')










