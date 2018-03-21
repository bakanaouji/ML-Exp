import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_scatter_3d(save_path, x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(np.reshape(x, x.size),
            np.reshape(y, y.size),
            np.reshape(z, z.size),
            "o")
    plt.savefig(save_path)
    plt.close()
