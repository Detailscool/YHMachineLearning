from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # [X, Y] = np.meshgrid(np.linspace(0.01,0.01,1),0.01:0.01:1)
