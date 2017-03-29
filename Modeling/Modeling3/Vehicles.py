#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'vehicles.csv'
    data = pd.read_csv(path)

    print data.columns
    # print pd.value_counts(data.fuelType)
    # print pd.value_counts(data.make)

    # grouped = data.groupby('year')
    # averageed = grouped['comb08','highway08','city08'].mean()
    # averageed.columns = ['comb08_mean', 'highway08_mean', 'city08_mean']
    # averageed['year'] = averageed.index
    # print averageed.head(5)
    #
    # x = np.arange(len(averageed.index))
    # y = averageed.comb08_mean
    # plt.figure(figsize=(12, 6))
    # plt.title(u'平均每加仑汽油可行驶英里数随时间的变化')
    # plt.xlabel(u'年数')
    # plt.ylabel(u'每加仑汽油可行驶英里数')
    # plt.xticks(x, averageed.year, rotation=45)
    # plt.plot(x, y, lw=2)
    # plt.scatter(x, y,s=50)
    # plt.grid()
    # plt.show()

    critera_1 = data.fuelType1.isin(['Regular Gasoline','Permium Gasoline','Midgrade Gasoline'])
    critera_2 = data.fuelType2.isnull()
    critera_3 = data.atvType != 'Hybrid'
    vehicles_non_hybrid = data[critera_1 & critera_2 & critera_3]
    print len(vehicles_non_hybrid)

    grouped = vehicles_non_hybrid.groupby(['year'])
    averageed = grouped['comb08'].mean()
    print averageed.head(5)

    # x = np.arange(len(averageed.index))
    # y = averageed
    # print y, type(y)
    # plt.figure(figsize=(12, 6))
    # plt.title(u'平均每加仑汽油可行驶英里数随时间的变化')
    # plt.xlabel(u'年数')
    # plt.ylabel(u'每加仑汽油可行驶英里数')
    # plt.xticks(x, averageed.index, rotation=45)
    # plt.plot(x, y, lw=2)
    # plt.scatter(x, y, s=50)
    # plt.grid()
    # plt.show()

    critera = vehicles_non_hybrid.displ.notnull()
    vehicles_non_hybrid = vehicles_non_hybrid[critera]
    vehicles_non_hybrid.displ = vehicles_non_hybrid.displ.astype('float')
    vehicles_non_hybrid.comb08 = vehicles_non_hybrid.comb08.astype('float')

    # x = vehicles_non_hybrid.displ
    # y = vehicles_non_hybrid.comb08
    # plt.figure(figsize=(12, 6))
    # plt.title(u'每加仑汽油可行驶英里数与引擎数的相关性')
    # plt.xlabel(u'引擎数')
    # plt.ylabel(u'每加仑汽油可行驶英里数')
    # plt.scatter(x, y, s=50)
    # plt.grid()
    # plt.show()

    grouped = vehicles_non_hybrid.groupby(['year'])
    averageed = grouped['comb08', 'displ'].mean()
    print averageed.head(5)

    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(hspace=0.5)
    for index, value in enumerate(averageed):
        x = np.arange(len(averageed.index))
        y = averageed[value]
        str = '%d1%d' % (len(averageed.columns), index+1)
        print str
        plt.subplot(int(str))
        plt.xlabel(u'年份')
        if index == 0:
            str = u'每加仑行驶英里数'
        else:
            str = u'引擎数'
        plt.ylabel(str)
        plt.xticks(x, averageed.index, rotation=45)
        plt.plot(x, y, lw=2)
        str = u'%s随时间的变化关系' % str
        plt.title(str)
        plt.scatter(x, y, s=50)
        plt.grid()
    plt.show()

    pd.unique(vehicles_non_hybrid.cylinders)
    vehicles_non_hybrid_4 = vehicles_non_hybrid[vehicles_non_hybrid.cylinders == 4]

    # groupby_year_4_cylinder = vehicles_non_hybrid_4.groupby(['year']).make.nunique()
    # print groupby_year_4_cylinder.tail(5)
    #
    # plt.figure(figsize=(12, 6))
    # plt.title(u'四缸汽车品牌数量变化')
    # plt.ylabel(u'品牌数')
    # plt.xlabel(u'年份')
    # plt.plot(groupby_year_4_cylinder)
    # plt.scatter(groupby_year_4_cylinder.index, groupby_year_4_cylinder, s=50)
    # plt.grid()
    # plt.show()

    # groupby_year_4_cylinder = vehicles_non_hybrid_4.groupby(['year'])
    # # print groupby_year_4_cylinder.head()
    # unique_make = []
    # for name, group in groupby_year_4_cylinder:
    #     # print pd.unique(group['make'])
    #     unique_make.append(set(pd.unique(group['make'])))
    #
    # unique_make = reduce(set.intersection, unique_make[:-1])
    #
    # print unique_make
    #
    # boolean_mask = []
    # for index, row in vehicles_non_hybrid_4.iterrows():
    #     make = row['make']
    #     boolean_mask.append(make in unique_make)
    # df_commom_makes= vehicles_non_hybrid_4[boolean_mask]
    # df_commom_makes_grouped = df_commom_makes.groupby(['year', 'make']).mean()
    # df_commom_makes_grouped = df_commom_makes_grouped.reset_index()
    # print df_commom_makes_grouped.head(10)
    #
    # n = 1
    # plt.figure(figsize=(18, 10))
    # plt.title(u'各年各汽车制造商品牌油耗情况')
    # plt.subplots_adjust(hspace=0.5)
    # for i in unique_make:
    #     datas = df_commom_makes_grouped[df_commom_makes_grouped.make == i]
    #     plt.subplot(4,3,n)
    #     plt.title(i)
    #     plt.xlabel(u'年份')
    #     plt.ylabel(u'每加仑汽油')
    #     plt.plot(datas.year, datas.comb08)
    #     plt.scatter(datas.year, datas.comb08)
    #     plt.grid()
    #     n += 1
    # plt.show()
