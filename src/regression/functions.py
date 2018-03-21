import numpy as np


class SphereFunction(object):
    def __init__(self, n):
        self.n = n

    def sample(self, data_num):
        x_data = np.random.uniform(-5.0, 5.0, [data_num, self.n, 1])
        y_data = np.sum(x_data ** 2, axis=1)
        return x_data, y_data
