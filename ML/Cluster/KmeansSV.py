#!/usr/bin/python
# -*- coding:utf-8 -*-
#  KmeansSV.py
#  Created by HenryLee on 2017/8/31.
#  Copyright © 2017年. All rights reserved.
#  Description : 轮廓系数

import numpy as np
from sklearn.cluster import KMeans

X = [
    [9670250, 1392358258],  # 中国
    [2980000, 1247923065],  # 印度
    [9629091, 317408015],  # 美国
    [8514877, 201032714],  # 巴西
    [377873, 127270000],  # 日本
    [7692024, 23540517],  # 澳大利亚
    [9984670, 34591000],  # 加拿大
    [17075400, 143551289],  # 俄罗斯
    [513115, 67041000],  # 泰国
    [181035, 14805358],  # 柬埔寨
    [99600, 50400000],  # 韩国
    [120538, 24052231]  # 朝鲜
    ]

X = np.array(X)

a = X[:, :1] / float(np.max(X[:, :1].ravel())) * 10000
b = X[:, 1:] / float(np.max(X[:, 1:].ravel())) * 10000
X = np.concatenate((a, b), axis=1)
# print X
# X = np.hstack((a, b))
# print X

n_clusters = 4

cls = KMeans(n_clusters=n_clusters).fit(X)

print cls.cluster_centers_
print cls.labels_

def manhattan_distance(x, y):
    return np.sum(abs(x-y))

distance_sum = 0
X0_label = cls.labels_[0]
X0_group = X[cls.labels_ == X0_label]
for v in X0_group:
    distance_sum += manhattan_distance(X[0], v)
av = distance_sum / len(X0_group)

print av

distance_sum = 0
for n in range(n_clusters):
    memebers = X[cls.labels_ == n]
    distance_min = 10000
    for v in memebers:
        if np.array_equal(X[0], v):
            continue
        distance = manhattan_distance(X[0], v)
        if distance < distance_min:
            distance_min = distance
    distance_sum += distance_min
bv = distance_sum / n_clusters

print bv

sv = float(bv - av)/max(av, bv)

print sv