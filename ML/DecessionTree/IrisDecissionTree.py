#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pydotplus

iris_features_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

def iris_type(s):
    iris = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return iris[s]

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = '10.iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    # print data
    x, y = np.split(data, (4,), axis=1)
    # print x
    # print y

    x = x[:, :2]
    # print x
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

    model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)

    # with open('iris.dot', 'w') as f:
    #     tree.export_graphviz(model, out_file=f)
    #
    # dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_features_E, class_names=iris_class,
    #                                 filled=True, rounded=True, special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf('iris.pdf')
    # f = open('iris.png', 'wb')
    # f.write(graph.create_png())
    # f.close()

    N, M = 50, 50
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1,t2)
    print x1
    print x2
    x_show = np.stack((x1.flat, x2.flat), axis=1)
    print 'x_show :', x_show, '\nshape : ', x_show.shape

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)
    # print y_show_hat.shape
    y_show_hat = y_show_hat.reshape(x1.shape)
    print y_show_hat.shape

    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=120, cmap=cm_dark, marker='*')
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(),edgecolors='k', s=40, cmap=cm_dark)
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
    plt.show()

    y_test = y_test.reshape(-1)
    depth = np.arange(1, 15)
    err_list = []
    criterion = ['gini', 'entropy']
    for d in depth:
        model = DecisionTreeClassifier(criterion=criterion[1], max_depth=d)
        model.fit(x_train,y_train)
        y_test_hat = model.predict(x_test)
        result = (y_test_hat == y_test)
        err = 1 - np.mean(result)
        err_list.append(err)
        print '深度: %d, 错误率: %.2f%%' % (d, err * 100)
    plt.figure(facecolor='w')
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel(u'深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()
