#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = r'../DecessionTree/10.iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    # print data
    x_prime, y = np.split(data, (4,), axis=1)
    # print x_prime
    # print y

    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    y = y.ravel()

    plt.figure(facecolor='w', figsize=(10, 9))

    for i, pair in enumerate(feature_pairs):
        x = x_prime[:, pair]

        clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=3)
        clf.fit(x, y)

        N, M = 50, 50
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)
        x_test = np.stack((x1.flat, x2.flat), axis=1)
        if i == 0:
            print 'x1 : ', x1, '\nx2 : ', x2, '\nx_test : ', x_test

        y_hat = clf.predict(x)
        c = np.count_nonzero(y_hat == y)

        print '特征：  ', iris_feature[pair[0]], ' + ', iris_feature[pair[1]],
        print '\t预测正确数目：', c,
        print '\t准确率: %.2f%%' % (100 * float(c) / float(len(y)))

        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

        y_hat = clf.predict(x_test)
        y_hat = y_hat.reshape(x1.shape)
        plt.subplot(2, 3, i + 1)
        plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
        plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=cm_dark)  # 样本
        plt.xlabel(iris_feature[pair[0]], fontsize=14)
        plt.ylabel(iris_feature[pair[1]], fontsize=14)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
    plt.tight_layout(2.5)
    plt.subplots_adjust(top=0.92)
    plt.suptitle(u'随机森林对鸢尾花数据的两特征组合的分类结果', fontsize=18)
    plt.show()
