import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    points = np.linspace(-1, 1, 4, endpoint=False)
    x, y = np.meshgrid(points, points)
    print 'x:\n', x
    print 'y:\n', y

    z = np.sqrt(x**2+y**2)
    print 'z:\n', z
    plt.imshow(z)
    plt.colorbar()
    plt.title('Image plot of $\sqrt{x^2+y^2}$ for a grid of values')
    plt.show()