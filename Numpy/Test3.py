import numpy as np

if __name__ == "__main__":
    arr = np.arange(24).reshape((2, 3, 4))
    print 'arr:\n', arr,'\n', arr.shape

    brr = arr.transpose((2, 1, 0))
    print 'brr:\n', brr,'\n', brr.shape



