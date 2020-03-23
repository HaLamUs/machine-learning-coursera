import numpy as np
import os
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D 

# Load data
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size # print(X.shape)

# # print out some data points
# print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
# print('-'*26)
# for i in range(10):
#     print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

# size_feature = [-5, 6, 9, 2, 4]
# mu = np.mean(size_feature)
# print(mu)
# sigma = np.std(size_feature)
# print(sigma)
# size_feature = [(x - mu)/sigma for x in size_feature]
# print(size_feature)
# print(np.mean(size_feature))
# # http://www.d.umn.edu/~deoka001/Normalization.html



def featureNormalize(X):
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    size_feature = X[:,0] # kích thước ngôi nhà 
    mu_size_feature = np.mean(size_feature)
    sigma_size_feature = np.std(size_feature)
    size_feature = [(x - mu_size_feature)/sigma_size_feature for x in size_feature]

    num_bed_feature = X[:,1]
    mu_num_bed_feature = np.mean(num_bed_feature)
    sigma_num_bed_feature = np.std(num_bed_feature)
    num_bed_feature = [(x - mu_num_bed_feature)/sigma_num_bed_feature for x in num_bed_feature]
    
    
    mu = [mu_size_feature, mu_num_bed_feature]
    sigma = [sigma_size_feature, sigma_num_bed_feature]
    
    X_norm = np.stack([size_feature, num_bed_feature], axis=1)
    # print(X_norm.shape)
    return X_norm, mu, sigma

# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

# print('Computed mean:', mu)
# print('Computed standard deviation:', sigma)

X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
# print(X.shape) print(X[0])

def computeCostMulti(X, y, theta):
    m = y.shape[0] # number of training examples
    J = 0
    J_sum = 0
    J_sum = np.dot(X, theta)
    J_sum = (J_sum - y)**2
    J = np.sum(J_sum) / (2*m)
    return J

J = computeCostMulti(X, y, theta=np.array([0.0, 0.0, 0.0]))
# print('With theta = [0, 0, 0] \nCost computed = %.2f' % J)

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    theta = theta.copy()
    J_history = [] 
    m = y.shape[0]
    for i in range(num_iters):
        tam = computeCostMulti(X, y, theta)
        J_history.append(tam)

        # theta_i = theta_i - (alpha/m) * sum (h(x) -y) * x(i)
        # h(x) = theta_0 + theta_1*X1 + theta_2*X2
        test = []
        theta_sum = 0
        theta_sum = np.dot(X, theta)
        theta_sum = theta_sum - y 
        for idx, val in enumerate(theta):
            test1 = np.dot(theta_sum, X[:, idx])
            test3 = (np.sum(test1) * alpha) / m
            theta_val = val - test3
            test.append(theta_val)
        theta = test 
    return theta, J_history

theta = np.zeros(3)
# some gradient descent settings
iterations = 500
alpha = 0.01
theta, J_history = gradientDescentMulti(X ,y, theta, alpha, iterations)
# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
# pyplot.show()
# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))


# mu = [mu_size_feature, mu_num_bed_feature]
#     sigma = [sigma_size_feature, sigma_num_bed_feature]
    
#     X_norm = np.stack([size_feature, num_bed_feature], axis=1)
#     # print(X_norm.shape)
#     return X_norm, mu, sigma [(x - mu_num_bed_feature)/sigma_num_bed_feature
# normalize 1650
size_1650_normal = (1650 - mu[0]) / sigma[0]
number_rooms_3_normal = (3 - mu[1]) / sigma[1]
# print('Size {0} number {1}'.format(size_1650_normal, number_rooms_3_normal))

price = np.dot([1, size_1650_normal, number_rooms_3_normal], theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))


########################### NORMAL EQUATIONS

data_v2 = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X_v2 = data_v2[:, :2]
y_v2 = data_v2[:, 2]
m_v2 = y_v2.size
X_v2 = np.concatenate([np.ones((m_v2, 1)), X], axis=1)
# print(np.zeros(X.shape[1]))
# print('X {0} T cua X {1}'.format(X[:, 1].shape, (X[:, 1].T).shape))

def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    test = []
    for idx, val in enumerate(theta):
        theta_test0 = np.stack([X[:,idx]], axis=1)
        theta_test1 = np.linalg.pinv(np.dot(theta_test0.T, theta_test0))
        theta_test2 = np.dot(theta_test1, theta_test0.T)
        theta_final = np.dot(theta_test2, y)
        test.append(theta_final[0])
    return test

# Calculate the parameters from the normal equation
theta_new = normalEqn(X, y)
# Display normal equation's result
print('Theta computed from the normal equations: {:s}'.format(str(theta_new)))

price = np.dot([1, size_1650_normal, number_rooms_3_normal], theta_new)

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))






