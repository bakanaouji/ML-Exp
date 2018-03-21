import numpy as np

from utils.data_plotting import plot_scatter_3d


class SphereFunction(object):
    def __init__(self, n):
        self.n = n

    def sample(self, data_num):
        x_data = np.random.uniform(-5.0, 5.0, [data_num, self.n, 1])
        y_data = np.sum(x_data ** 2, axis=1)
        return x_data, y_data


if __name__ == '__main__':
    func = SphereFunction(2)
    x_data, y_data = func.sample(1000)
    plot_scatter_3d('../../data/scatter.pdf', x_data[:, 0], x_data[:, 1], y_data)
