from numpy.random.mtrand import randn
import numpy as np

if __name__ == "__main__":
    arr = randn(8)
    print 'arr : ', arr

    arr.sort()
    print 'arr : ', arr

    brr = randn(2,3,4)
    # brr, y = np.meshgrid(np.arange(4) + 1,[1,1,1,1])
    print 'brr: \n',  brr

    brr.sort(0)
    print 'brr: \n', brr