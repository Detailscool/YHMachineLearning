#!/usr/bin/python
# -*- coding:utf-8 -*-
#  GrandientLr.py
#  Created by HenryLee on 2018/8/23.
#  Copyright © 2018年. All rights reserved.
#  Description :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

path = 'LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# 查看头部
print pdData.head()
# 查看矩阵的shape
print pdData.shape

positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]

# plt.figure(figsize=(400, 300))
plt.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
plt.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
plt.legend()
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()

# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# model
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)

    return grad


STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type, value, threshold):
    # 设定三种不同的停止策略
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


pdData.insert(0, 'Ones', 1)
# 获取特征矩阵和标签矩阵
orig_data = pdData.values
cols = orig_data.shape[1]
X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]
# 初始化参数矩阵
theta = np.zeros([1, 3])
print theta


# 洗牌，每次梯度下降取样本前要把数据集的顺序打乱
def shuffleData(data):
    # 随机排序函数shuffle
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


# 参数迭代更新
def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad
        costs.append(cost(X, y, theta))  # 保存损失值
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad

        if stopCriterion(stopType, value, thresh):
            break

    return theta, i - 1, costs, grad, time.time() - init_time


# 此处的代码是将迭代的过程以图表的形式展示
def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)

    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)

    name += strDescType + " descent - Stop: "

    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)

    name += strStop
    print "***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur)

    # plt.figure(figsize=(200, 150))
    plt.plot(np.arange(len(costs)), costs, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(name.upper() + ' - Error vs. Iteration')
    plt.show()

    return theta


n = 100
runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
# runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)