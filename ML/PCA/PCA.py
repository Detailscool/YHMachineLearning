#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print '%s正确率：%.3f%%' % (tip, acc_rate)
    return acc_rate

if __name__ == '__main__':
    data_train = pd.read_csv('optdigits.tra', header=None)
    data_test = pd.read_csv('optdigits.tes', header=None)

    X_train = data_train[np.arange(64)]
    y_train = data_train[64]

    estimator = PCA(n_components=2) #将高维度特征向量（六十四维）压缩到两个维度的PCA
    X_pca = estimator.fit_transform(X_train)

    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i, color in enumerate(colors):
        px = X_pca[:, 0][y_train.as_matrix() == i]
        py = X_pca[:, 1][y_train.as_matrix() == i]
        plt.scatter(px, py, c=color)
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

    X_test = data_test[np.arange(64)]
    y_test = data_test[64]

    np.random.seed(0)

    svc = LinearSVC()
    svc.fit(X_train, y_train)
    y_hat = svc.predict(X_test)
    show_accuracy(y_hat, y_test, 'LinearSVC')

    estimator = PCA(n_components=20)
    X_pca_train = estimator.fit_transform(X_train)
    X_pca_test = estimator.transform(X_test)

    pca_svc = LinearSVC()
    pca_svc.fit(X_pca_train, y_train)
    y_pca_hat = pca_svc.predict(X_pca_test)
    show_accuracy(y_pca_hat, y_test, 'PCA')