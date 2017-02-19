import numpy as np

if __name__ == "__main__":
    arr = np.random.rand(4, 4)
    print 'arr:\n', arr

    brr = np.where(arr > 0.5, 2, arr)
    print 'brr:\n', brr,'\n', brr.dtype
