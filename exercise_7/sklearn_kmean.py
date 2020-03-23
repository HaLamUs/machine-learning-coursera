import os
# import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn.cluster import KMeans

# ======= Experiment with these parameters ================
K = 6

A = mpl.image.imread(os.path.join('Data', 'logo_truong.png'))
A /= 255
print(A.shape)

X = A.reshape(-1, 3)
print(X.shape)

kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
# print(kmeans.cluster_centers_)
print(kmeans.labels_)
centroids = kmeans.cluster_centers_
idx = kmeans.labels_








# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
# Reshape the recovered image into proper dimensions
X_recovered = centroids[idx, :].reshape(A.shape)


# Display the original image, rescale back by 255
fig, ax = pyplot.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(A*255)
ax[0].set_title('Ảnh gốc')
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow(X_recovered*255)
ax[1].set_title('Ảnh đã nén với %d cụm' % K)
ax[1].grid(False)
pyplot.show()