import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative

from pylab import scatter, show, title, xlabel, ylabel, plot, contour

a = np.identity(5)


data=np.loadtxt('ex1data1.txt',delimiter=',')
plt.scatter(data[:,0],data[:,1])
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population in 10,000s')
# plt.show()

# ======= format the data =======
x = data[:,0]
y = data[:,1]
y_data = y

# Number of examples
m = len(y)

y = reshape(y,(m,1))
x_vectorized = ones(shape=(m, 2))
x_vectorized[:, 1] = x

# Gradient Descent
theta = zeros(shape=(2, 1))
alpha = 0.01
iterations = 1500

# this works, but doesn't actually impact the gradient descent function
def cost_function(theta, x, y):
    prediction_y = dot(x, theta)
    J = (1.0 / (2*m)) * dot(transpose(prediction_y - y) , (prediction_y - y))
    print(J, 'cost')
    return J

def gradient_descent(theta, x_vectorized, y, alpha, iterations):
    for i in range(iterations):
        prediction_y = dot(x_vectorized, theta)
        theta = theta - alpha * (1.0/m) * dot(transpose(x_vectorized), (prediction_y - y))
        # print(theta[0], '== theta ZERO', theta[1], '== theta ONE', ' after ' +str(i+1), 'iterations', '\n')
    return theta

theta = gradient_descent(theta, x_vectorized, y, alpha, iterations)
print(theta)

# Visualizing the outcome
def plot_solution(x, y, solution):
    # create figure and axes
    fig = plt.figure()
    # split the page into a 1x1 array of subplots and put solution in the first one (111)
    ax = fig.add_subplot(111)

    # plots scatter for x, y
    ax.scatter(x, y, color='red', marker='o', s=100)
    # plots solution
    ax.plot(x, solution, color='green')
    plt.show()
    
plot_solution(x, y_data, dot(x_vectorized,theta))

