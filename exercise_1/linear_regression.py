import numpy as np
import os
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D 

def warmUpExercise():
    A = np.eye(5)
    return A

A = warmUpExercise()# print(A.shape)

# Read comma separated data
data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size  #print(X.size) print(X.shape) print(y.shape[0])

def plotData(x, y):
    fig = pyplot.figure()  # open a new figure
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population of City in 10,000s')
    # pyplot.show()

# plotData(X, y)
X = np.stack([np.ones(m), X], axis=1)

def computeCost(X, y, theta):
    # initialize some useful values
    m = y.size  # number of training examples
    J_sum = 0
    theta_0 = theta[0]
    theta_1 = theta[1]
    feature_0 = X[:,0]
    feature_1 = X[:,1]
    J = 0
    # J = 1/2m * Sum (the_ta0 + the_ta1 - y)^2
    
    for idx, val in enumerate(y):
        J_sum += (theta_0*feature_0[idx] + theta_1*feature_1[idx] - val)**2
    J = J_sum / (2*m)
    return J

def computeCost2(X, y, theta):
    m = y.size  # number of training examples
    J = 0
    J_sum = 0
    for idx, val in enumerate(theta):
        J_sum += np.dot(val, X[:,idx])
    J_sum = (J_sum - y)**2
    J = np.sum(J_sum) / (2*m)
    return J

def computeCost3(X, y, theta):
    m = y.size  # number of training examples
    J = 0
    J_sum = 0
    # for idx, val in enumerate(theta):
    #     J_sum += np.dot(val, X[:,idx])
    J_sum = np.dot(X, theta)
    # J_sum = np.dot(theta, X) # đéo đc 
    J_sum = (J_sum - y)**2
    J = np.sum(J_sum) / (2*m)
    return J

J = computeCost3(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

# further testing of the cost function
J = computeCost3(X, y, theta=np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 54.24')

def computeTheta(X, y, theta, alpha):
    theta_0 = theta[0]
    theta_1 = theta[1]
    feature_0 = X[:,0]
    feature_1 = X[:,1]
    m = y.shape[0]
    # theta_0 = theta_0 - alpha * (1/2m) * sum (theta_0 + theta_1*x(i) - y(i))
    theta_0_update = 0
    theta_1_update = 0
    theta_0_sum = 0
    theta_1_sum = 0
    for idx, val in enumerate(y):
        theta_0_sum += (theta_0*feature_0[idx] + theta_1*feature_1[idx] - val)
        theta_1_sum += (theta_0*feature_0[idx] + theta_1*feature_1[idx] - val)*feature_1[idx]

    theta_0_update = theta_0 - ((alpha*theta_0_sum)/m)
    theta_1_update = theta_1 - ((alpha*theta_1_sum)/m)

    return theta_0_update, theta_1_update

def gradientDescent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        # save the cost J in every iteration
        tam = computeCost(X, y, theta)
        J_history.append(tam)
        theta_0, theta_1 = computeTheta(X, y, theta, alpha)
        theta = [theta_0, theta_1]
        # print(tam)
    
    return theta, J_history

def gradientDescent2(X, y, theta, alpha, num_iters):
    theta = theta.copy()
    J_history = [] 
    m = y.shape[0]
    for i in range(num_iters):
        tam = computeCost2(X, y, theta)
        J_history.append(tam)

        # theta_i = theta_i - (alpha/m) * sum (h(x) -y) * x(i)
        # h(x) = theta_0 + theta_1*X1 + theta_2*X2
        test = []
        theta_sum = 0
        for idx, val in enumerate(theta):
             theta_sum += np.dot(val, X[:, idx]) # là h(x)
        theta_sum = theta_sum - y # là h(x) - y
        for idx, val in enumerate(theta):
            test1 = np.dot(theta_sum, X[:, idx]) # nhân cho x 
            test3 = (np.sum(test1) * alpha) / m
            theta_val = val - test3
            test.append(theta_val)
        theta = test
    return theta, J_history

theta = np.zeros(2)
# some gradient descent settings
iterations = 1500
alpha = 0.01
theta, J_history = gradientDescent2(X ,y, theta, alpha, iterations)
# print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
# print('Expected theta values (approximately): [-3.6303, 1.1664]')

# # plot the linear fit
# plotData(X[:, 1], y)
# pyplot.plot(X[:, 1], np.dot(X, theta), '-')
# pyplot.legend(['Training data', 'Linear regression']);
# pyplot.show()

# # Predict values for population sizes of 35,000 and 70,000
# predict1 = np.dot([1, 3.5], theta) # h(x) = theta_0 + theta_1*x
# print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))

# predict2 = np.dot([1, 7], theta)
# print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))

# # grid over which we will calculate J
# theta0_vals = np.linspace(-10, 10, 100)
# theta1_vals = np.linspace(-1, 4, 100)

# # initialize J_vals to a matrix of 0's
# J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# # Fill out J_vals
# for i, theta0 in enumerate(theta0_vals):
#     for j, theta1 in enumerate(theta1_vals):
#         J_vals[i, j] = computeCost(X, y, [theta0, theta1])
        
# # Because of the way meshgrids work in the surf command, we need to
# # transpose J_vals before calling surf, or else the axes will be flipped
# J_vals = J_vals.T

# # surface plot
# # fig = pyplot.figure(figsize=(12, 5))
# # ax = fig.add_subplot(121, projection='3d')
# # ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
# # pyplot.xlabel('theta0')
# # pyplot.ylabel('theta1')
# # pyplot.title('Surface')


# # contour plot
# # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
# ax = pyplot.subplot(122)
# pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
# pyplot.xlabel('theta0')
# pyplot.ylabel('theta1')
# pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
# pyplot.title('Contour, showing minimum')
# # pyplot.show()


