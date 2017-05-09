#!/usr/bin/python
# -*- coding:utf-8 -*-

import xgboost as xgb
import numpy as np

def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    label = y.get_label()
    g = p - label
    h = p*(1.0-p)
    return g, h

def custom_loss(y_hat, y): #别人的自定义损失函数
    label = y.get_label()
    penalty = 2.0
    grad = -label/y_hat + penalty*(1 - label)/(1 - y_hat) #梯度
    hess = label/y_hat**2 + penalty*(1 - label)/(1 - y_hat)**2 #2阶导
    return grad, hess

def error_rate(y_hat, y):
    # print 'y_getlabel : ', y.get_label()
    return 'error', float(sum(y.get_label() != (y_hat>0.5))) / len(y_hat)

if __name__ == '__main__':
    data_test = xgb.DMatrix('14.agaricus_test.txt')
    data_train = xgb.DMatrix('14.agaricus_train.txt')
    print data_train.get_label()

    params = {'max_depth': 4, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    # params = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    bst = xgb.train(params=params, dtrain=data_train, num_boost_round=3, evals=watchlist, obj=custom_loss, feval=error_rate)
    # bst = xgb.train(params=params, dtrain=data_train, num_boost_round=3, evals=watchlist, obj=log_reg,
    #                 feval=error_rate)

    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print y_hat
    print y

    error = sum(y != (y_hat > 0.5))
    error_rate = float(error)/len(y_hat)
    print '样本总数：', len(y_hat)
    print '错误个数 ： %4d' % error
    print '正确率 ： %.6f%%' % (100 * (1 - error_rate))