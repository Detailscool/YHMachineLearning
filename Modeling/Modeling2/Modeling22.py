#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def drawCurve(a,CurrentLevel):
    RegentLevel = []
    for i in range(30):
        RegentLevel.append(CurrentLevel)
        CurrentLevel = (a * CurrentLevel - a * (CurrentLevel ** 2))
    plt.hist(RegentLevel, bins=100)

if __name__ == '__main__':
    # a = 1.5
    a = [1.5, 2.5, 3, 3.4, 3.5, 3.8]
    for index, d in enumerate(a):
        # str = '%d%d%d' % (len(a), int(np.floor(np.sqrt(len(a)))), index + 1)
        plt.subplot(int('%d%d%d' % (len(a), int(np.floor(np.sqrt(len(a)))), index + 1)))
        for i in range(1, 100):
            drawCurve(d, i/100.0)
        str = 'a = %s' % d
        plt.annotate(str, xy=(0.15, 5), xytext=(0.25, 6), arrowprops=dict(facecolor='black', shrink=0.1))
    plt.grid()
    plt.show()