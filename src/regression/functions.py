import numpy as np


class FunctionBase(object):
    def __init__(self, n, output_dim):
        self.n = n
        self.output_dim = output_dim

    def sample(self, data_num):
        pass

    def input_dimension(self):
        return self.n

    def output_dimension(self):
        return self.output_dim


class SphereFunction(FunctionBase):
    def __init__(self, n):
        super().__init__(n, 1)

    def sample(self, data_num):
        x_data = np.random.uniform(-5.0, 5.0, [data_num, self.n])
        y_data = np.sum(x_data ** 2, axis=1)
        y_data = y_data.reshape([y_data.size, 1])
        return x_data, y_data
