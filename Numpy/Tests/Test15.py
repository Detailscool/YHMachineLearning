# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    time = [i for i in range(0, 19)]
    number = [9.6, 18.3, 29, 47.2, 71.1, 119.1, 174.6, 257.3,
              350.7, 441.0, 513.3, 559.7, 594.8, 629.4, 640.8,
              651.1, 655.9, 659.6, 661.8]
    f = np.polyfit(time, number, 3)
    y = np.polyval(f, time)  # 根据拟合之后的函数来求函数值
    plt.plot(time, y, color='b')  # 根据函数值画图并设定颜色
    plt.title('Relationship between time and number')
    plt.xlabel('time')
    plt.ylabel('number')
    plt.plot(time, number, color='r')
    plt.show()