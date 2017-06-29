import numpy as np

a = np.arange(1, 13).reshape(3, 4)
b = np.arange(13, 25).reshape(2, -1)

x, y = np.meshgrid(a, b)

print 'x:\n', x, '\n', x.shape
print 'y:\n', y, '\n', y.shape
print 'np.c_ : \n', np.c_[x, y]
print 'stack : \n', np.stack((x.flat, y.flat), axis=1)