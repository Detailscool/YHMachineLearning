#!/usr/bin/python
# -*- coding:utf-8 -*-
#  Test18.py
#  Created by HenryLee on 2017/9/10.
#  Copyright © 2017年. All rights reserved.
#  Description :

from sklearn.datasets import load_sample_images
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import corner_harris

plt.axis('off')

dataset = load_sample_images()
img = dataset.images[0]
print img.shape
# plt.imshow(img)
# plt.show()

harris_coords = corner_harris(img)
print harris_coords.shape
y, x = np.transpose(harris_coords)

plt.imshow(img)
plt.scatter(x, y ,s=30, c='r')
plt.show()
