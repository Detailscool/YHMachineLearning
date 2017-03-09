import numpy as np
from numpy.linalg import *

if __name__ == '__main__':
    x = np.linspace(1, 12, 12).reshape((3, 4))
    y = np.linspace(13, 24, 12).reshape((4, 3))
    dot = x.dot(y)
    print 'dot : \n', dot

    inv = inv(dot)
    print 'inv : \n', inv

    det = det(dot)
    print 'det : ', det

    feature, vector = eig(dot)
    print 'feature : ', feature,'\nvector : ',vector

    q, r = qr(dot)
    print 'q : ', q, '\nr : ', r

    # trace = trace(dot)

    b = np.array([[6], [6], [6]])
    solve = solve(dot, b)
    print 'solve : ', solve






