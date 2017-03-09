#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint

if __name__ == "__main__":
    path = 'day.csv'

    # 1.手写读取
    # f = file(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     # d = map(float, d.split(','))
    #     d = d.split(',')
    #     # print type(d), d
    #     x.append(d[1 : -1])
    #     y.append(d[-1])
    # pprint(x)
    # pprint(y)

    #2.自带库读取
    # f = file (path, 'rb')
    # # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()

    #3.numpy读取
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print p

    #4.pandas读取
    data = pd.read_csv(path)
    date = data['dteday']
    x = data[['season', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']]
    y = data['cnt']
    # print type(x)
    # print x
    # print y

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12,9))
    plt.subplot(331)
    plt.plot(data['season'], y, 'ro')
    plt.title('season')
    plt.grid()
    plt.subplot(332)
    plt.plot(data['weekday'], y, 'g^')
    plt.title('weekday')
    plt.grid()
    plt.subplot(333)
    plt.plot(data['workingday'], y, 'b*')
    plt.title('workingday')
    plt.grid()
    plt.subplot(334)
    plt.plot(data['weathersit'], y, 'b*')
    plt.title('weathersit')
    plt.grid()
    plt.subplot(335)
    plt.plot(data['temp'], y, 'b*')
    plt.title('temp')
    plt.grid()
    plt.subplot(336)
    plt.plot(data['atemp'], y, 'b*')
    plt.title('atemp')
    plt.grid()
    plt.subplot(337)
    plt.plot(data['hum'], y, 'b*')
    plt.title('hum')
    plt.grid()
    plt.subplot(338)
    plt.plot(data['windspeed'], y, 'b*')
    plt.title('windspeed')
    plt.grid()
    plt.subplot(339)
    plt.plot(data['casual'], y, 'b*')
    plt.title('casual')
    plt.grid()
    # plt.tight_layout()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print 'x : ', x_test, x_train
    print 'y : ', y_test, y_train
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print 'model :', model
    print linreg.coef_
    print linreg.intercept_

    y_hat = linreg.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)
    rmse = np.sqrt(mse)
    print 'mse : ', mse, 'rmse : ',rmse

    t = np.arange(len(x_test))
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
