#!/usr/bin/python
# -*- coding:utf-8 -*-
#  fetch_olivett_faces.py
#  Created by HenryLee on 2017/9/15.
#  Copyright © 2017年. All rights reserved.
#  Description :

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
faces = dataset.data

plt.axis('off')

def ploy_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2* n_col, 2.6*n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i+1)
        vmax = max(comp.max(), -comp.min())

        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest', vmin=-vmax, vmax=vmax)
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0)

estimators = [('PCA', PCA(n_components=6, whiten=True)),
              ('NMF', NMF(n_components=6, init='nndsvda', tol=5e-3))]

for name, exstimator in estimators:
    exstimator.fit(faces)
    components_ = exstimator.components_
    ploy_gallery(name, components_[:n_components])

plt.show()
