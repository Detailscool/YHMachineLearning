#!/usr/bin/python
# -*- coding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    path = '14.wine.data'
    data = np.loadtxt(path, dtype=float, delimiter=',')
    y, x = np.split(data, (1, ), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

    lr = LogisticRegression()
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    print 'Logistic回归正确率 : ', float(sum(y_hat.ravel() == y_test.ravel()))/len(y_hat)

    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    params = {'max_depth': 4, 'eta': 0.1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=6, evals=watchlist)
    y_hat = bst.predict(dtest)
    print 'XGBoost正确率 : ', float(sum(y_hat.ravel() == y_test.ravel())) / len(y_hat)





