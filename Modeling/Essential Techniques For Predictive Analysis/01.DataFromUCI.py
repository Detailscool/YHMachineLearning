#!/usr/bin/python
# -*- coding:utf-8 -*-

import httplib
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'

connection = httplib.HTTPSConnection("archive.ics.uci.edu")
connection.request('GET', url)
response = connection.getresponse()
print 'status :', response.status, '\n', 'reason :', response.reason
data = response.read()
# print 'data :', type(data), ' \n', data

xList = []

data = data.split('\n')[:-1]
for line in data:
    xList.append(line.split(','))

print 'row :', len(xList), ', column :', len(xList[0])
# print xList

type = [0]*3
type_count = []

for row in xList:
    for num in row:
        try:
            a = float(num)
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(num) > 0:
                type[1] += 1
            else:
                type[2] += 1
    type_count.append(type)
    type = [0]*3

# data = pd.DataFrame(type_count, columns=['Number', 'Strings', 'Others'])
data = np.array(type_count)
# print 'data : \n', data

x, y = np.split(xList, (60, ), axis=1)
print x
print y

x = np.array(x, dtype=float)
# x = x[:, 3]

print 'Mean : ', np.mean(x), 'Standard Deviation : ', np.std(x)

ntiles = 4
percent_boundary = []
for i in range(ntiles+1):
    percent_boundary.append(np.percentile(x, i*100/ntiles))
print 'Boundaries for 10 Equal Percentiles : ', percent_boundary

x = np.unique(x)
print x, len(x)

# cat_dict = dict(zip(x.tolist(), range(len(x))))
# print cat_dict

value_count_result = pd.Series(y.ravel()).value_counts()
print 'value_count_result : \n', value_count_result



