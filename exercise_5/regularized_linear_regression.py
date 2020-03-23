import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

# Load from ex5data1.mat, where all variables will be store in a dictionary
data = loadmat(os.path.join('Data', 'ex5data1.mat'))

# Extract train, test, validation data from dictionary
# and also convert y's form 2-D matrix (MATLAB format) to a numpy vector

X, y = data['X'], data['y'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]

# m = Number of examples
m = y.size #print(m)

# Plot training data
# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.show()


def linearRegCostFunction(X, y, theta, lambda_=0.0):
    m = y.size # number of training examples
    J = 0
    grad = np.zeros(theta.shape)
    # print(X.shape) # print(theta.shape) # print(y.shape)

    #====================== H_1, H_2 just a step

    theta_left = np.delete(theta, 0) 
    H_1 = np.dot(X, theta)
    H_2 = (H_1 - y)**2
    J_1 = (np.sum(H_2)) / (2*m)
    const_val = lambda_ / (2*m)
    reg_J = np.sum(theta_left**2) 
    J_2 = const_val*reg_J
    J = J_1 + J_2

    grad_1 = np.dot((H_1 - y), X)
    grad_1 = grad_1 / m
    grad_2 = (theta_left*lambda_) / m
    grad_2 = np.insert(grad_2, 0, 0) 
    grad = grad_1 + grad_2

    return J, grad

# theta = np.array([1, 1])
# J, _ = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

# print('Cost at theta = [1, 1]:\t   %f ' % J)
# print('This value should be about 303.993192)\n' % J)

# theta = np.array([1, 1])
# J, grad = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

# print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}] '.format(*grad))
# print(' (this value should be about [-15.303016, 598.250744])\n')

# # add a columns of ones for the y-intercept
# X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
# theta = utils.trainLinearReg(linearRegCostFunction, X_aug, y, lambda_=0)

# #  Plot fit over the data
# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1.5)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.plot(X, np.dot(X_aug, theta), '--', lw=2);
# pyplot.show()


def learningCurve(X, y, Xval, yval, lambda_=0):
    m = y.size # Number of training examples
    error_train = np.zeros(m)
    error_val   = np.zeros(m)
    print(X.shape)
    print(y.shape)

    for i in range(1, m+1):
        # findTheta at first i examples in train data set 
        X_train_temp = X[:i]#X[:i, :]
        # print(X_train_temp.shape)
        y_train_temp = y[:i]
        theta_i = utils.trainLinearReg(linearRegCostFunction, 
        X_train_temp, y_train_temp, lambda_)

        # J_train_err_i, _ = linearRegCostFunction(X_train_temp, y_train_temp,
        # theta_i, lambda_)
        J_train_err_i, _ = linearRegCostFunction(X_train_temp, y_train_temp, 
        theta_i)
        J_val_err_i, _ = linearRegCostFunction(Xval, yval, 
        theta_i)
        error_train[i-1] = J_train_err_i
        error_val[i-1] = J_val_err_i

    return error_train, error_val

# X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
# Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
# error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)
# print(error_train.shape) 
# print(error_val.shape) #print(error_train)print(error_val)

# pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
# pyplot.title('Learning curve for linear regression')
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 150])
# pyplot.show()

# print('# Training Examples\tTrain Error\tCross Validation Error')
# for i in range(m):
#     print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))
# err_val min là 33.512228 ==> high bias 

# for i in range(2, 8+1):
#     print(i)

def polyFeatures(X, p):
    X_poly = X#np.zeros((X.shape[0], p))
    for i in range(2, p+1):
        X_at_i = X**i
        X_poly = np.hstack((X_poly, X_at_i))
    return X_poly

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
# print(X_poly.shape) 
# print(X_poly[:,1]) 
# print(X)
X_poly, mu, sigma = utils.featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
# print(X_poly_test.shape) 
# print(X_poly_test[:,1]) 
# print(Xtest)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)


# lambda_ = 1

# # theta = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
# J, _ = linearRegCostFunction(X_poly, y, theta, 1)
# print('Cost at theta = [1, 1]:\t   %f ' % J)


# theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y,
#                              lambda_=lambda_, maxiter=55)

# # Plot training data and fit
# pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

# utils.plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)

# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
# pyplot.ylim([-20, 50])

# pyplot.figure()
# error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
# pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)

# pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 100])
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.show()

# print('Polynomial Regression (lambda = %f)\n' % lambda_)
# print('# Training Examples\tTrain Error\tCross Validation Error')
# for i in range(m):
#     print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

def validationCurve(X, y, Xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    for i in range(len(lambda_vec)):
        # print(i)
        theta_lamda_i = utils.trainLinearReg(linearRegCostFunction, 
        X, y, lambda_vec[i])
        J_train_err_i, _ = linearRegCostFunction(X, y, 
        theta_lamda_i) 
        J_val_err_i, _ = linearRegCostFunction(Xval, yval, 
        theta_lamda_i) # print(J_val_err_i)
        error_train[i-1] = J_train_err_i
        error_val[i-1] = J_val_err_i

    return lambda_vec, error_train, error_val

# lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)
# print(error_train.shape)
# pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.xlabel('lambda')
# pyplot.ylabel('Error')
# pyplot.show()

# print('lambda\t\tTrain Error\tValidation Error')
# for i in range(len(lambda_vec)):
#     print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))

# Optional 1

theta_with_lamda_3 = utils.trainLinearReg(linearRegCostFunction, 
        X_poly, y, 3)
J_test_err, _ = linearRegCostFunction(X_poly_test, ytest, theta_with_lamda_3)
print(J_test_err)


# Optional 2
# the training error and cross validation error for i examples


# lambda_val value for this step
lambda_val = 0.01

# number of iterations
times = 50

# initialize error matrices
error_train_rand = np.zeros((m, times))
error_val_rand   = np.zeros((m, times))
# print(error_train_rand.shape)


for i in range(1,m+1):
    for k in range(times):
         # choose i random training examples
        rand_sample_train = np.random.permutation(X_poly.shape[0])
        rand_sample_train = rand_sample_train[:i]

        # choose i random cross validation examples
        rand_sample_val   = np.random.permutation(X_poly_val.shape[0])
        rand_sample_val   = rand_sample_val[:i]

         # define training and cross validation sets for this loop
        X_poly_train_rand = X_poly[rand_sample_train,:]
        y_train_rand      = y[rand_sample_train]
        X_poly_val_rand   = X_poly_val[rand_sample_val,:]
        yval_rand         = yval[rand_sample_val]

        theta_lamda_i = utils.trainLinearReg(linearRegCostFunction, 
        X_poly_train_rand, y_train_rand, lambda_val)
        J_train_err_i, _ = linearRegCostFunction(X_poly_train_rand, y_train_rand, theta_lamda_i) 
        J_val_err_i, _ = linearRegCostFunction(X_poly_val_rand, yval_rand, theta_lamda_i) 

error_train = np.mean(error_train_rand, axis=1)
error_val   = np.mean(error_val_rand, axis=1)

# error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)
# print(error_train.shape) 
# print(error_val.shape) #print(error_train)print(error_val)

pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
pyplot.title('Learning curve for linear regression')
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('Number of training examples')
pyplot.ylabel('Error')
pyplot.axis([0, 13, 0, 150])
pyplot.show()

