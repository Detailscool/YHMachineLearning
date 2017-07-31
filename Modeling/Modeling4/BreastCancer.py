#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

colum_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Unifomity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print '%s正确率：%.3f%%' % (tip, acc_rate)
    return acc_rate

if __name__ == '__main__':
    path = 'breast-cancer-wisconsin.data'
    data = pd.read_csv(path, names=colum_name)

    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna(how='any')
    print data, '\n', data.shape

    x_train, x_test, y_train, y_test = train_test_split(data[colum_name[1: 10]], data[colum_name[10]], test_size=0.25, random_state=33)
    print y_train.value_counts()
    print y_test.value_counts()

    print x_train.info()

    ss = StandardScaler()
    X_train = ss.fit_transform(x_train)
    X_test = ss.transform(x_test)

    lr = LogisticRegression(random_state=1)
    sgdc = SGDClassifier(random_state=1)
    lr.fit(X_train, y_train)
    lr_y_hat = lr.predict(X_test)
    sgdc.fit(X_train, y_train)
    sgdc_y_hat = sgdc.predict(X_test)

    a = show_accuracy(lr_y_hat, y_test, 'lr')
    b = show_accuracy(sgdc_y_hat, y_test, 'SGDC')