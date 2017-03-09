import numpy as np
# import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    theta1,  theta2 = 0, 0
    i = np.arange(1, 7)
    i2 = i ** 2
    alpha = 0.01
    beta = 0.1
    a = np.zeros(6)
    for _ in range(5000):
        a = -np.arange(1, 7) * theta2 - theta1 - 1
        a = np.exp(a)
        f1 = np.dot(a, i)
        f2 = np.sum(a)
        theta1 += alpha * (f1 - np.e) * f1 + beta * (f2 - 1) * f2
    # plt.show()
    # show(a/a.sum())