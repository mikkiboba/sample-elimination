import numpy as np
import matplotlib.pyplot as plt

# Number of points
n_points = 5000

# generate random coordinates for the points
x = np.random.rand(n_points)
y = np.random.rand(n_points)

# plot the points
plt.scatter(x,y,n_points/1000)
plt.savefig('./pictures/random_sampling.png', dpi=500)
plt.show()
