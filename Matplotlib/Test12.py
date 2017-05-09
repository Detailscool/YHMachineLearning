#!/usr/bin/python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    plt.figure(figsize=(9, 6))
    n = 1000
    #rand 均匀分布和 randn高斯分布
    x = np.random.randn(1, n)
    print 'x : \n', np.sort(x)
    y = np.random.randn(1, n)
    print 'y : \n', y
    T = np.arctan2(x, y)
    print 'T : \n', T
    plt.scatter(x, y, c=T, s=25, alpha=0.4, marker='o')
    #T:散点的颜色
    #s：散点的大小
    #alpha:是透明程度
    plt.show()