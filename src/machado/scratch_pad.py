import numpy as np
x_i = np.array([5.5, 6.2])
print(x_i.shape)
theta = np.zeros((x_i.shape[0] + 1, 1))
x_i = np.insert(x_i, 0, 1)
x_i = x_i.reshape(-1,1)
x_i = np.dot(x_i.T, theta)
print(x_i)