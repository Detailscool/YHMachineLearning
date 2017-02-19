import numpy as np

if __name__ == '__main__':
    nums = np.array([1, 1, 1, 2, 3, 4, 4, 5, 5, 6])
    arr = np.unique(nums)
    print 'arr : ', arr

    brr = np.array([2, 3, 7, 8, 4])
    crr = np.intersect1d(arr, brr)
    print 'crr : ', crr

    print 'arr contains crr :',np.in1d(arr, crr)

    drr = np.union1d(arr, brr)
    print 'drr : ', drr

    err = np.setdiff1d(brr, arr)
    print 'err : ', err

    frr = np.setxor1d(arr, brr)
    print 'frr : ', frr

