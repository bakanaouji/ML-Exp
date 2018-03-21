from regression.functions import SphereFunction
from utils.data_plotting import plot_scatter_3d


def main():
    func = SphereFunction(2)
    x_data, y_data = func.sample(1000)
    plot_scatter_3d('../data/scatter.pdf', x_data[:, 0], x_data[:, 1], y_data)


if __name__ == '__main__':
    main()
