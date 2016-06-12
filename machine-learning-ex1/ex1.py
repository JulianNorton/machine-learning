import numpy as np
from matplotlib import pyplot as plt
# import numpy as np
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
# import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

a = np.identity(5)

# print(a)

data=np.loadtxt('ex1data1.txt',delimiter=',')

# print(data)

data=np.loadtxt('ex1data1.txt',delimiter=',')
plt.scatter(data[:,0],data[:,1])
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population in 10,000s')
# plt.show()

# ======= format the data =======
x=data[:,0]
Y=data[:,1]
m=len(Y)
y=reshape(Y,(m,1))

X = ones(shape=(m, 2))
X[:, 1] = x

# ======= define the cost function J for linear regression =======
def cost_function(theta, X, y):
        prediction=dot(X,theta)
        J = (1.0/(2*m))*dot(transpose(prediction-y),(prediction-y))               
        return J

    
# ======= Gradient Descent =======
# define function for batch gradient descent
def gradient_descent(theta, X, y, alpha, num_iters):
    m=len(y)
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):
        theta=theta-(alpha/m)*(dot(transpose(X),(dot(X,theta)-y)))
        J_history[i] = cost_function(theta, X, y)
    return theta, J_history

# set parameters for gradient descent
alpha = 0.01
iterations = 1500
theta = zeros(shape=(2, 1)) 

#compute and display initial cost
print('Initial cost: '+str(cost_function(theta,X,y)[0][0]))

# perform gradient descent 
theta, J_history = gradient_descent(theta, X, y, alpha, iterations)
print('Theta found by gradient descent: '+str(theta[0][0])+', '+str(theta[1][0]))

# visualise outcome
def plot_solution(x,y,solution):
    # create figure and axes
    fig = plt.figure()
    # split the page into a 1x1 array of subplots and put solution in the first one (111)
    ax = fig.add_subplot(111)

    # plots scatter for x, y
    ax.scatter(x, y, color='red', marker='o', s=100)
    # plots solution
    ax.plot(x, solution, color='green')
    plt.show()
plot_solution(x,Y,dot(X,theta))

# ======= visualise cost function J over a grid of values for theta =======
theta0_vals=linspace(-10,10,100);
theta1_vals=linspace(-1,4,100);
J_vals=zeros(shape=(len(theta0_vals),len(theta1_vals)));

for i, elementi in enumerate(theta0_vals):
    for j, elementj in enumerate(theta1_vals):
        thetatest=zeros(shape=(2,1))
        thetatest=[[elementi],[elementj]]
        J_vals[i][j]= cost_function(thetatest, X, y)

# We need to transpose J_vals before plotting it
J_vals=transpose(J_vals)

contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel(r'$\theta_0$')
ylabel(r'$\theta_1$')
show()