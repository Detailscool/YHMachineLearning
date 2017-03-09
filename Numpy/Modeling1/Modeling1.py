#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '酵母菌.csv'
    data = pd.read_csv(path)
    print data

    time = data['time']
    amount = data['amount']
    # plt.plot(time, amount)
    # plt.xlabel('time')
    # plt.ylabel('amount')
    # plt.show()

    f = np.polyfit(time, amount, 3)
    y = np.polyval(f, time)
    plt.plot(time, y, 'b')
    plt.title('Relationship between time and amount')
    plt.xlabel('time')
    plt.ylabel('amount')
    plt.plot(time, amount, 'r')
    plt.show()

    delta = data['delta']
    factor = amount * (665 - amount)
    f = np.polyfit(factor, delta, 1)
    print f


