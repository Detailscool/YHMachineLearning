from random import normalvariate
import numpy as np
import time

if __name__ == '__main__':
    # samples = np.random.normal(size=(4,4))
    N = 1000000
    start = time.clock()
    samples = [normalvariate(0, 1) for _ in xrange(N)]
    print 'time1 : ', time.clock() - start

    start = time.clock()
    samples = np.random.normal(size=N)
    print 'time2 : ', time.clock() - start
