import numpy as np

if __name__ == "__main__":
    arr = np.empty((8, 4))
    for i in range(8):
        arr[i] = i
    print 'arr:\n', arr

    brr = np.tile(range(8), [4,1])
    print 'brr:\n', brr

    crr = np.stack(arr,axis=1)
    print 'crr:\n', crr
