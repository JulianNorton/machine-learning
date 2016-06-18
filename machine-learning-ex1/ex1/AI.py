import numpy as np
from matplotlib import pyplot as plt
from numpy import zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

data=np.loadtxt('ex1data1.txt',delimiter=',')
x = data[:,0]
y = data[:,1]
m = len(y) 
plt.scatter(x,y)
plt.ylabel("profit in 10k")
plt.xlabel("population in 10k")
plt.title("joijoijoi")
y_vectorized = reshape(y,(m,1))               
x_vectorized = ones(shape=(m,2))
x_vectorized[:,1] = x
theta = zeros(shape=(2,1))

def cost_function(theta, x, y): 
	guess_Y = dot(x, theta)
	J = (1.0 / (2*m))*dot(transpose(guess_Y - y) , (guess_Y - y))	
	return J

iterations = 1000
alpha = 0.01 #ALPHAAAAAA

def gradient_decsent(theta, x_vectorized, y, alpha, iterations):
	for i in range(iterations):
		guess_Y = dot(x_vectorized, theta)	
		theta = theta - alpha * (1.0/m) * dot(transpose(x_vectorized), (guess_Y - y))
		#print(theta[0], '== theta ZERO', theta[1], '== theta ONE', ' after ' +str(i+1), 'iterations', '\n')
	return(theta)
	#print(cost_function(theta, x, y))       #stufff
	#print("theta is =", theta)
	#return theta

theta = gradient_decsent(theta, x_vectorized, y, alpha, iterations)

plt.show()
# # visualise outcome
def plot_solution(x,y,solution):
#     # create figure and axes
	fig = plt.figure()
#     # split the page into a 1x1 array of subplots and put solution in the first one (111)
	ax = fig.add_subplot(111)
#     # plots scatter for x, y
	ax.scatter(x, y, color='red', marker='o', s=100)
#     # plots solution
	ax.plot(x, solution, color='green')
	plt.show()
print(theta)
plot_solution(x,y, dot(x_vectorized, theta))