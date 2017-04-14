#!/usr/bin/python
# -*- coding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

def iris_type(s):
    type = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return type[s]

if __name__ == '__main__':
    path = u'../DecessionTree/10.iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

    dtest = xgb.DMatrix(x_test, label=y_test)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    pramas = {'max_depth': 4, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params=pramas, dtrain=dtrain, num_boost_round=3, evals=watchlist)
    y_hat = bst.predict(dtest)

    print y_test.shape, '-', y_hat.shape, 'type:', type(y_hat)
    result = y_test.reshape(y_hat.shape) == y_hat

    print y_test.reshape(y_hat.shape)
    print y_hat
    print '样本个数 ：', len(y)
    print '正确率 ： %.6f%%' % (float(sum(result))/len(y_hat) * 100)
