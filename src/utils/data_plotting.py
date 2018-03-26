import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_data(save_path, x_min, x_max, y_min, y_max):
    data = pd.DataFrame.from_csv(save_path)
    N = len(data.columns)
    div = 256
    save_path = save_path.replace('.csv', '')
    for i in range(int(math.ceil(float(N / div)))):
        d = (data.ix[:, i * div:(i + 1) * div])
        plt.plot(d)
        plt.grid()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(save_path + '_' + str(i) + '.pdf')
        plt.close()


def plot_scatter_3d(save_path, x, y, z):
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(np.reshape(x, x.size),
            np.reshape(y, y.size),
            np.reshape(z, z.size),
            "o")
    plt.savefig(save_path)
    plt.close()
