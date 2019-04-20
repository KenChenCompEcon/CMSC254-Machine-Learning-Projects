"""
CMSC 25400 - Homework 2
Instructor: Prof. Kondor
Author: Ken Chen
"""

import numpy as np
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.n, self.d = data.shape
      
    def distort(self, assign, centers):
        s = 0
        # Our distortion function will take the form of squared distance summation
        for cluster in range(self.k):
            df = self.data[assign == cluster]
            tot = ((df-centers[cluster])**2).sum(axis = 1)
            s += tot.sum()
        return s
    
    def converge(self, centers_old, centers_new):
        assign = np.full(self.n, -1)
        loss = []
        while (centers_new != centers_old).any():
            centers_old = centers_new.copy()
            # Assign each point to the closet center
            for i in range(self.n):
                dists = ((centers_old - self.data[i])**2).sum(axis = 1)
                cluster = np.argmin(dists)
                assign[i] = cluster
            # Calculate the new center
            for cluster in np.unique(assign):
                new = self.data[assign == cluster].mean(axis = 0)
                centers_new[cluster] = new
            # Calculate distort function
            loss.append(self.distort(assign, centers_new))
        return assign, centers_new, loss
    
    def kmeans(self, seed = 1):
        centers_old = np.full(shape = [self.k, self.d], fill_value = -1) 
        
        np.random.seed(seed)
        centers_new = self.data[np.random.choice(self.n, self.k, replace = False)]
        return self.converge(centers_old, centers_new)
        
    def kmeansPlus(self, seed=1):
        centers_old = np.full(shape = [self.k, self.d], fill_value = -1) 
        # Choose the initial centers
        np.random.seed(seed)
        centers_new = self.data[np.random.choice(self.n, 1)]
        for i in range(1, self.k):
            d_mat = np.full((self.n, i), -1.0)
            for j in range(i):
                d = ((self.data - centers_new[j])**2).sum(axis = 1)
                d_mat[:,j] = d
            dists = d_mat.min(axis = 1)
            p = dists/dists.sum()
            center = self.data[np.random.choice(self.n, 1, p = p)]
            centers_new = np.vstack((centers_new, center))

        return self.converge(centers_old, centers_new)

# loads the data in a numpy matrix
data = np.loadtxt('mnist_small.txt')
print("Shape of the data: {}".format(data.shape))
data_norm = data/255

# Try the kmeans algorithm
m = Kmeans(data_norm, 10)
res = m.kmeans(1234)

# visualize the all the images
plt.gray()
for i in range(1,11):
    plt.subplot(2,5,i)
    plt.imshow(data[res[0] == (i-1)].mean(axis = 0).reshape(8,8))
plt.show()

# plot the distortion function against iteration
plt.figure()
for seed in np.arange(100,120):
    res = m.kmeans(seed)
    plt.plot(np.arange(1, len(res[2])+1), res[2])
plt.xlabel("#Iteration"); plt.ylabel("Distortion Function")
plt.title("Distortion against Iteration Times: by Kmeans")
plt.grid()
plt.show()


# Try the kmeansPlus algorithm
res_2 = m.kmeansPlus(1234)

# visualize the all the images
plt.gray()
for i in range(1,11):
    plt.subplot(2,5,i)
    plt.imshow(data[res_2[0] == (i-1)].mean(axis = 0).reshape(8,8))
plt.show()

# plot the distortion function against iteration
plt.figure()
for seed in np.arange(100,120):
    res = m.kmeansPlus(seed)
    plt.plot(np.arange(1, len(res[2])+1), res[2])
plt.xlabel("#Iteration"); plt.ylabel("Distortion Function")
plt.title("Distortion against Iteration Times: by KmeansPlus")
plt.grid()
plt.show()
