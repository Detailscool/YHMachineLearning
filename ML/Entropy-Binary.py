import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p = np.linspace(0, 1, 1001)
    he = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
    plt.plot(p, he)
    plt.show()
