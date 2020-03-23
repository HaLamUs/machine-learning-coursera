import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex4data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

# Number of training examples
m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

# utils.displayData(sel)

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the weights into variables Theta1 and Theta2
weights = loadmat(os.path.join('Data', 'ex4weights.mat'))

# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)
# print(Theta1.shape) print(Theta2.shape) print(Theta2[:,0]) print(Theta2[0,:])

# Unroll parameters 
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
# print(nn_params.shape) # 401*25 + 26*10 = 10285
#print(y.shape) print(y[4999])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def vector_y_ith(label, num_labels):
    tam = np.zeros(num_labels)
    tam[label] = 1
    return tam

def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = sigmoid(z)*(1 - sigmoid(z))
    return g

 
def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    W = np.zeros((L_out, 1 + L_in))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

# test = np.arange(9).reshape((3,3))
# print(test)
# # test = test[:,1]
# # test[:,1] *= 0
# ssss = test[:,1:]
# print("======")
# print(ssss)

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # print(Theta1.shape); print(Theta2.shape)

    # Setup some useful variables
    m = y.size
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    # print(Theta1_grad.shape); print(Theta2_grad.shape)  print(Theta1_grad[0])

    X = np.concatenate([np.ones((m, 1)), X], axis=1) #print(X.shape) print(X[0].shape)
    J_sum = 0
    delta_capital = 0

    for i in range(m):
        a_1 = X[i]
        z_2 = np.dot(Theta1, a_1)
        a_2 = sigmoid(z_2) #print(a_2.shape)
        a_2 = np.insert(a_2, 0, 1) 
        z_3 = np.dot(Theta2, a_2)
        a_3 = sigmoid(z_3) #print(a_3.shape)
        y_k = vector_y_ith(y[i], num_labels) 
        J_sum += np.dot(y_k.T, np.log(a_3)) + np.dot((1 - y_k).T, np.log(1 - a_3)) 
        delta_3 = a_3 - y_k
        z_2 = np.insert(z_2, 0, 1) 
        delta_2 = np.dot(Theta2.T, delta_3) * sigmoidGradient(z_2) #print(delta_2.shape)
        delta_2 = delta_2[1:]
        # sss = a_1[None,:] sssss = delta_2[None,:] print(sss.shape) print(sssss.shape)
        Theta1_grad = Theta1_grad +  a_1[None,:] * delta_2[None,:].T 
        Theta2_grad = Theta2_grad + a_2[None,:] * delta_3[None,:].T

    J_sum = J_sum / (-m) 
    J_reg = (np.sum(nn_params**2)*lambda_) / (2*m)
    J = J_sum + J_reg
    Theta1_grad /= m
    Theta2_grad /= m
    
    
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + Theta2[:,1:]*lambda_ / m
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + Theta1[:,1:]*lambda_ / m


    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

lambda_ = 0
J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
print('The cost should be about                   : 0.287629.')

# Weight regularization parameter (we set this to 1 here).
# lambda_ = 1
# J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
#                       num_labels, X, y, lambda_)

# print('Cost at parameters (loaded from ex4weights): %.6f' % J)
# print('This value should be about                 : 0.383770.')



z = np.array([-99, -0.5, 0, 0.5, 1000])
g = sigmoidGradient(z)
# print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
# print(g)

# print('Initializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
# print(initial_Theta1.shape) print(initial_Theta2.shape)
# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

# utils.checkNNGradients(nnCostFunction)

#  Check gradients by running checkNNGradients
lambda_ = 3
utils.checkNNGradients(nnCostFunction, lambda_)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X, y, lambda_)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))
print('(for lambda = 3, this value should be about 0.576051)')

#  After you have completed the assignment, change the maxiter to a larger
#  value to see how more training helps.
options= {'maxiter': 100}

# #  You should also try different values of lambda
# lambda_ = 1

# # Create "short hand" for the cost function to be minimized
# costFunction = lambda p: nnCostFunction(p, input_layer_size,
#                                         hidden_layer_size,
#                                         num_labels, X, y, lambda_)

# # Now, costFunction is a function that takes in only one argument
# # (the neural network parameters)
# res = optimize.minimize(costFunction,
#                         initial_nn_params,
#                         jac=True,
#                         method='TNC',
#                         options=options)

# # get the solution of the optimization
# nn_params = res.x
        
# # Obtain Theta1 and Theta2 back from nn_params
# Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
#                     (hidden_layer_size, (input_layer_size + 1)))

# Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
#                     (num_labels, (hidden_layer_size + 1)))

# pred = utils.predict(Theta1, Theta2, X)
# print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

# utils.displayData(Theta1[:, 1:])
# pyplot.show()
