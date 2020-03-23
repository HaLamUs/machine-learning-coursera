import os
import numpy as np
# Import regular expressions to process emails
import re

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# from IPython.display import HTML, display, clear_output

# try:
#     pyplot.rcParams["animation.html"] = "jshtml"
# except ValueError:
#     pyplot.rcParams["animation.html"] = "html5"

from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

def findClosestCentroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)
 

    # print(centroids.shape)
    # print(centroids[:,np.newaxis].shape)
    ds = centroids[:,np.newaxis]-X
    # print(ds.shape)
    e_dists =  np.sqrt(np.sum(np.square(ds),axis=2)) #-1
    # print(e_dists.shape) # sum lại nó mất 1 chiều rồi 
    idx = np.argmin(e_dists, axis=0) # lấy min theo cột đây

    # for index, x in enumerate(X):
    #     tam = []
    #     for k in centroids:
    #         min = euclideanDistance(x, k)
    #         tam.append(min)
    #     idx_min = np.argmin(tam)
    #     idx[index] = idx_min
    
    return idx

def euclideanDistance(x, k):
    return np.sqrt(np.sum((x - k)**2))

# Load an example dataset that we will be using
data = loadmat(os.path.join('Data', 'ex7data2.mat'))
X = data['X']

# Select an initial set of centroids
K = 3   # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples:')
print(idx[:3])
print('(the closest centroids should be 0, 2, 1 respectively)')


def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    # ver 1
    clusters = [X[idx==ci] for ci in range(K)]
    # print(clusters)
    # centroids = np.asarray([(1/len(cl))*np.sum(cl, axis=0) for cl in clusters if len(cl)>0])
    centroids = np.asarray([np.sum(cl, axis=0) / (len(cl)) for cl in clusters if len(cl)>0])

    # ver 2
    # for k in range(K):
    #     indices = [index_x for index_x, x in enumerate(idx) if x == k]
    #     len_x = len(X[indices])
    #     mean_x = (np.sum(X[indices], axis=0)) / len_x 
    #     centroids[k] = mean_x
    # print(centroids)
    
    # ver 3
    # for k in range(K):
    #     tong = 0
    #     count = 0 
    #     for index, x in enumerate(X):
    #         if (idx[index] == k):
    #             tong += x
    #             count += 1
    #     print(tong)
    #     print(count)
    #     centroids[k] = tong/count 
             

    return centroids

# Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

# x_1 = X[0]
# x_2 = X[1]
# print(x_1)
# print(x_2)
# tama = 0.5 * (x_1 + x_2)
# print(tama)

print('Centroids computed after initial finding of closest centroids:')
print(centroids)
print('\nThe centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')

# test = np.zeros((300, 3))
# test[:, 0] = np.ones(300)
# print(test[:,0])


# # Load an example dataset
# data = loadmat(os.path.join('Data', 'ex7data2.mat'))

# # Settings for running K-Means
# K = 3
# max_iters = 10

# # For consistency, here we set centroids to specific values
# # but in practice you want to generate them automatically, such as by
# # settings them to be random examples (as can be seen in
# # kMeansInitCentroids).
# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])


# # Run K-Means algorithm. The 'true' at the end tells our function to plot
# # the progress of K-Means
# centroids, idx, anim = utils.runkMeans(X, initial_centroids,
#                                        findClosestCentroids, computeCentroids, max_iters, True)
# anim

def kMeansInitCentroids(X, K):
    m, n = X.shape
    
    centroids = np.zeros((K, n))
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    return centroids

# ======= Experiment with these parameters ================
# You should try different values for those parameters
K = 16
max_iters = 10

# Load an image of a bird
# Change the file name and path to experiment with your own images
A = mpl.image.imread(os.path.join('Data', 'bird_small.png'))
# ==========================================================

# Divide by 255 so that all values are in the range 0 - 1
A /= 255
print(A.shape)

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(-1, 3)
print(X.shape)

# When using K-Means, it is important to randomly initialize centroids
# You should complete the code in kMeansInitCentroids above before proceeding
initial_centroids = kMeansInitCentroids(X, K)
print(initial_centroids.shape)

# Run K-Means
centroids, idx = utils.runkMeans(X, initial_centroids,
                                 findClosestCentroids,
                                 computeCentroids,
                                 max_iters)
print(centroids.shape)
print(idx.shape)

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
# Reshape the recovered image into proper dimensions
X_recovered = centroids[idx, :].reshape(A.shape)


# Display the original image, rescale back by 255
fig, ax = pyplot.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(A*255)
ax[0].set_title('Original')
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed, with %d colors' % K)
ax[1].grid(False)
pyplot.show()