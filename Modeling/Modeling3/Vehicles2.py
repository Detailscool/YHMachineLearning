#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'vehicles.csv'
    data = pd.read_csv(path)

    data = data[(data['make'] == 'Toyota')]

    grouped = data.groupby(['model', 'year'])

    unique_make = []
    for name, group in grouped:
        # print name[0]
        if name[0] not in unique_make :
            unique_make.append(name[0])

    print unique_make

    averageed = grouped['comb08', 'highway08', 'city08'].mean()
    averageed = averageed.reset_index()
    print averageed, type(averageed)

    plt.figure(figsize=(12, 20),facecolor='w')
    plt.subplots_adjust(hspace=0.8, top=0.95)
    count = 6
    n = 0
    for index, value in enumerate(unique_make):
        if n < count:
            datas = averageed[averageed.model == value]
            if len(datas) > 20:
                x = np.arange(len(datas.year))
                plt.subplot(count, 2, n+1)
                plt.title(value)
                plt.xlabel(u'年份')
                plt.ylabel(u'每加仑汽油')
                plt.plot(x, datas.comb08)
                # plt.ployfit
                plt.scatter(x, datas.comb08)
                f = np.polyfit(x, datas.comb08, 9)
                y = np.polyval(f, x)
                plt.plot(x, y, color='r')
                plt.xticks(x, datas.year, rotation=90)
                plt.grid()
                n += 1
    plt.tight_layout(2)
    plt.show()




