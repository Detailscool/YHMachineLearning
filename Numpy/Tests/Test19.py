#!/usr/bin/python
# -*- coding:utf-8 -*-
#  Test19.py
#  Created by HenryLee on 2018/8/31.
#  Copyright © 2018年. All rights reserved.
#  Description :

import numpy as np
from sklearn.model_selection import KFold

data = np.arange(24)
b = np.arange(start=24) + 24

fold = KFold(5).split(data, y=b)
for iteration, indices in enumerate(fold):
    print('iteration : %s , indices : %s' % (iteration, indices))



