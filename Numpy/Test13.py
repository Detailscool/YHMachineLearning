#coding:utf8
from numpy.random import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nstep = 100
    draws = randint(0, 2, size=nstep)
    print '\ndraws : ', draws
    steps = np.where(draws>0, 1, -1)
    walks = steps.cumsum()
    x = np.array(range(nstep))

    print '\nwalks : ',walks
    print 'min :', walks.min(),'\nmax : ', walks.max()

    plt.plot(x, walks)
    plt.legend()
    plt.show()
