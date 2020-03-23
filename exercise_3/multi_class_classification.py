import os
import numpy as np
from matplotlib import pyplot
# Optimization module in scipy
from scipy import optimize
# will be used to load MATLAB mat datafile format
from scipy.io import loadmat


# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size
# print(X.shape)


def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

# displayData(sel)
# pyplot.show()

# test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularization parameter
lambda_t = 3

def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))

def lrCostFunction(theta, X, y, lambda_):
    #Initialize some useful values
    m = y.size
    
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    h_0 = sigmoid(np.dot(X, theta))
    y_negative = np.dot(-1, y)
    J_sum = y_negative*np.log(h_0) - (1 - y)*np.log(1 - h_0)
    J = np.sum(J_sum) / m
    J = J + ((np.sum(theta**2) * lambda_) / (2*m))
    test = []
    grad_sum = h_0 - y
    tam = np.dot(X.T, grad_sum)
    tam = tam / m
    tam = tam + ((theta*lambda_) / m)
    grad = tam
    
    return J, grad

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

# print('Cost         : {:.6f}'.format(J))
# print('Expected cost: 2.534819')
# print('-----------------------')
# print('Gradients:')
# print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
# print('Expected gradients:')
# print(' [0.146561, -0.548558, 0.724722, 1.398003]')

def oneVsAll(X, y, num_labels, lambda_):
    # Some useful variables
    m, n = X.shape
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))
    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    for c in range(num_labels):
        initial_theta = np.zeros(n + 1)
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction, 
                                initial_theta, 
                                (X, (y == c), lambda_), 
                                jac=True, 
                                method='TNC',
                                options=options) 
        # cost = res.fun 
        # theta = res.x
        all_theta[c] = res.x

    return all_theta

# all_theta = np.zeros((num_labels,  5))
# test = np.ones(5)
# print(test)
# for c in range(num_labels):
#     all_theta[c] = test
# print(all_theta)

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)
# print((all_theta[2])[:5]) # print(all_theta[4][:5])
# print(y[-5:]) print(num_labels) print(y[5:])

def predict(theta, X):
    m = X.shape[0] # Number of training examples
    return sigmoid(np.dot(X, theta))

def predictOneVsAll(all_theta, X):
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(m)
    A = np.zeros((num_labels, m))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    for c in range(num_labels):
        A[c] = predict(all_theta[c], X)
    print(A.shape)
    for idx in range(m): # khúc này làm hơi ngc dòng và cột 
        # print(A[:,idx].shape)
        predict_label = np.argmax(A[:,idx])
        # print(predict_label)
        p[idx] = predict_label
    return p

# print(all_theta.shape)
# print(X.shape)
pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))

# X = np.concatenate([np.ones((m, 1)), X], axis=1)
# tam = predict(all_theta[3], X)
# quang = np.array([[1, 2, 3, 4, 3], [13, 24, 33, 44, 443]])
# print(quang[:,4])
# print(np.argmax(tam))
# print(tam[:5])
# for c in range(num_labels):
    # print(c)
