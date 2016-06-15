import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative

from pylab import scatter, show, title, xlabel, ylabel, plot, contour

a = np.identity(5)

# print(a)

# data=np.loadtxt('ex1data1.txt',delimiter=',')

# print(data)

data=np.loadtxt('ex1data1.txt',delimiter=',')
plt.scatter(data[:,0],data[:,1])
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population in 10,000s')
# plt.show()

# ======= format the data =======
x = data[:,0]
y = data[:,1]
m = len(y)

y = reshape(y,(m,1))
reshaping_x = ones(shape=(m, 2))
reshaping_x[:, 1] = x
x = reshaping_x

# Gradient Descent
theta = zeros(shape=(2, 1))
alpha = 0.01
iterations = 1500

def cost_function(theta, x, y):
    prediction = dot(x, theta)
    J = (1.0 / (2*m)) * dot(transpose(prediction - y) , (prediction - y))
    return J


print(cost_function(theta, x, y))


# z = ones(shape=(m, 2))
# x=[]
# X[:, 1] = z
# X[:, 1] = x
# X = ones(shape=(m, 2))
# X[:, 1] = x
