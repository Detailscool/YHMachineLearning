import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

if __name__ == '__main__':
    x = np.linspace(1., 4., 2)
    y = np.linspace(1., 2., 2)
    x, y = np.meshgrid(x, y)
    z = np.array([[0.5, 1.5], [2.5, 3.5]])
    cmap = mpl.colors.ListedColormap(['r', 'y', 'g', 'b'])
    bounds = [ 0.,  1.,  2.,  3.,  4.]
    ticks = [ 0.5, 1.5, 2.5, 3.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N )
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(z)
    cbar = fig.colorbar(m, norm=norm, boundaries=bounds, aspect=20, ticks=ticks)
    cbar.set_ticklabels(['A', 'B', 'C', 'D'])
    plt.show()