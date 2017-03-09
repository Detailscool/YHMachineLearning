import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # points = np.linspace(-1, 1, 4, endpoint=False)
    points = np.array([1, 2, 3, 4])
    x, y = np.meshgrid(points, [4, 4, 2, 3, 5])
    print 'x:\n', x
    print 'y:\n', y

    # z = np.sqrt(x**2+y**2)
    # print 'z:\n', z
    # plt.imshow(z)
    # plt.colorbar()
    # plt.title('Image plot of $\sqrt{x^2+y^2}$ for a grid of values')
    # plt.show()