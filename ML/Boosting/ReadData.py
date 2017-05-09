#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

def read_data(path):
    f = open(path)
    y = []
    for d in f:
        d = d.strip().split()
        y.append(d[1:])
    # y = np.array(y)
    # y = np.stack(y, axis=1)
    print len(y)


if __name__ == '__main__':
    path = '14.agaricus_train.txt'
    read_data(path)