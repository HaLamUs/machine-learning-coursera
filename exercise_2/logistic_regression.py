import os
import numpy as np
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize
import math

# Load data
# The first two columns contains the exam scores and the third column
# contains the label.
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

def plotData(X, y):
    # Create New Figure
    fig = pyplot.figure()
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

# plotData(X, y)
# pyplot.xlabel('Exam 1 score')
# pyplot.ylabel('Exam 2 score')
# pyplot.legend(['Admitted', 'Not admitted'])
# # pyplot.show()

# A = np.eye(5)
# z = np.array(A)
# print(np.dot(z, -1))

# sigmoid nó tính theo element-wise
def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z) #print(z.shape)
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)
    z_negative = np.dot(z, -1)

    g = 1 / (1 + np.power(math.e, z_negative))
    return g

# # Test the implementation of sigmoid function here
# z = 0
# z = [0.2, 0.4, 0.1]
# z = [[0.2, 0.4], [0.5, 0.7], [0.9, 0.004]]
# g = sigmoid(z)

# print('g(', z, ') = ', g)

# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def costFunction(theta, X, y):
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    h_0 = sigmoid(np.dot(X, theta))
    y_negative = np.dot(-1, y)
    J_sum = y_negative*np.log(h_0) - (1 - y)*np.log(1 - h_0)
    J = np.sum(J_sum) / m
    test = []
    grad_sum = h_0 - y
    for idx, val in enumerate(theta):
        grad1 = np.dot(grad_sum, X[:, idx])
        grad2 = np.sum(grad1) / m
        test.append(grad2)
    grad = test

    return J, grad

# no loop version
def costFunction2(theta, X, y):
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    h_0 = sigmoid(np.dot(X, theta))
    y_negative = np.dot(-1, y)
    J_sum = y_negative*np.log(h_0) - (1 - y)*np.log(1 - h_0)
    J = np.sum(J_sum) / m
    grad_sum = h_0 - y
    tam = np.dot(X.T, grad_sum)
    tam = tam / m
    grad = tam
    return J, grad

# yy = np.eye(5)#np.array([1, 2])
# tam = 1 - yy #np.dot(-1, yy)
# print(y) # print(np.log(5))

# Initialize fitting parameters
initial_theta = np.zeros(n+1)

# cost, grad = costFunction2(initial_theta, X, y)

# print('Cost at initial theta (zeros): {:.3f}'.format(cost))
# print('Expected cost (approx): 0.693\n')

# print('Gradient at initial theta (zeros):')
# print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
# print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
# test_theta = np.array([-24, 0.2, 0.2])
# cost, grad = costFunction(test_theta, X, y)

# print('Cost at test theta: {:.3f}'.format(cost))
# print('Expected cost (approx): 0.218\n')

# print('Gradient at test theta:')
# print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
# print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')

# set options for optimize.minimize
options= {'maxiter': 400}

# see documention for scipy's optimize.minimize  for description about
# the different parameters
# The function returns an object `OptimizeResult`
# We use truncated Newton algorithm for optimization which is 
# equivalent to MATLAB's fminunc
# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
res = optimize.minimize(costFunction,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x

# # Print theta to screen
# print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
# print('Expected cost (approx): 0.203\n');

# print('theta:')
# print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
# print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

def plotDecisionBoundary(plotData, theta, X, y):
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        pyplot.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        pyplot.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        pyplot.xlim([30, 100])
        pyplot.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
        pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)

# plotDecisionBoundary(plotData, theta, X, y)
# pyplot.show()

def predict(theta, X):
    m = X.shape[0] # Number of training examples
    # You need to return the following variables correctly
    p = np.zeros(m) #print(p.shape)
    test = []
    h_0 = sigmoid(np.dot(X, theta)) #print(h_0.shape)
    for val in h_0:
        if val >= 0.5:
            test.append(1)
        else:
            test.append(0)
    p = np.array(test)
    return p

# #  Predict probability for a student with score 45 on exam 1 
# #  and score 85 on exam 2 
# prob = sigmoid(np.dot([1, 45, 85], theta))
# print('For a student with scores 45 and 85,'
#       'we predict an admission probability of {:.3f}'.format(prob))
# print('Expected value: 0.775 +/- 0.002\n')
# # print(np.array([1, 2, 3, 4]).shape)

# # Compute accuracy on our training set
# p = predict(theta, X)
# print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
# print('Expected accuracy (approx): 89.00 %')


# ===================== REGULARIZATION

# Load Data
# The first two columns contains the X values and the third column
# contains the label (y).
data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

# plotData(X, y)
# # Labels and Legend
# pyplot.xlabel('Microchip Test 1')
# pyplot.ylabel('Microchip Test 2')

# # Specified in plot order
# pyplot.legend(['y = 1', 'y = 0'], loc='upper right') # pyplot.show()

# tạo feature mới = polynomial

def mapFeature(X1, X2, degree=6):
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:, 0], X[:, 1])

def costFunctionReg(theta, X, y, lambda_):
    m = y.size  # number of training examples
    J = 0
    grad = np.zeros(theta.shape)

    J_part1, grad_1 = costFunction(theta, X, y)
    J_part2 = (np.sum(theta**2) * lambda_) / (2*m)
    J = J_part1 + J_part2

    test = []
    for idx, val in enumerate(grad_1):
        test1 = val + ((theta[idx]*lambda_) / m)
        test.append(test1)
    grad = test

    return J, grad

def costFunctionReg2(theta, X, y, lambda_):
    m = y.size  # number of training examples
    J = 0
    grad = np.zeros(theta.shape)

    J_part1, grad_1 = costFunction2(theta, X, y)
    J_part2 = (np.sum(theta**2) * lambda_) / (2*m)
    J = J_part1 + J_part2
    tam = grad_1 + ((theta*lambda_) / m)
    grad = tam
    return J, grad

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg2(initial_theta, X, y, lambda_)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg2(test_theta, X, y, 10)

print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

# ======================== VẼ HÌNH 

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lambda_ = 0

# set options for optimize.minimize
options= {'maxiter': 100}

res = optimize.minimize(costFunctionReg,
                        initial_theta,
                        (X, y, lambda_),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of OptimizeResult object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property of the result
theta = res.x

plotDecisionBoundary(plotData, theta, X, y)
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')
pyplot.legend(['y = 1', 'y = 0'])
pyplot.grid(False)
pyplot.title('lambda = %0.2f' % lambda_)
# pyplot.show()

# Compute accuracy on our training set
p = predict(theta, X)

# test = 0 == 1 
# # print(test) print(p[:10]) print(y[:10])

print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')


