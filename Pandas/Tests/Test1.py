#!/usr/bin/python
# -*- coding:utf-8 -*-
#  Test1.py
#  Created by HenryLee on 2018/8/27.
#  Copyright © 2018年. All rights reserved.
#  Description :


import pandas as pd

# pd.set_option('display.width=800')

data = pd.read_csv('../../ML/Regression/day.csv')
print(data)
a = data['casual'].idxmax(axis=1)
print(data.loc[a]['instant'])
data.drop()
# print(data)
# print(data['2'].idxmax(axis=1))