import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# import knn
from utils import DataSystem

def plot_dist(dist_list):
    gs1 = gridspec.GridSpec(4,2)
    axs = []
    fig = plt.figure()
    count = 0

    for dist in dist_list:
        data_class = DataSystem(2, 1000, dist)
        data = data_class.generate(True, None)

        axs.append(fig.add_subplot(gs1[count]))
        axs[-1].scatter(data[:,0], data[:, 1], s = 5)
        count+=1
        # plt.scatter(data[:,0], data[:,1])
    plt.show()

if __name__ == "__main__":
    dist_list = ['normal', 'uniform', 'geometric', 'gamma', 'beta', 'exponential', 'poisson']
    plot_dist(dist_list)