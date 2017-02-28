import numpy as np

if __name__ == '__main__':
    # arrays = [np.linspace(1, 12, 12).reshape(3,4) for _ in range(0,2)]
    # print 'origin : \n', arrays
    # print '1 : \n', np.stack(arrays, axis=0)
    # print '2 : \n', np.stack(arrays, axis=1)
    # print '3 : \n', np.stack(arrays, axis=2)

    # a = np.array([1, 2, 3])
    # b = np.array([2, 3, 4])
    # print np.stack((a, b))
    # print np.stack((a, b), axis=-1)

    a = np.array((1, 2, 3))
    b = np.array((2, 3, 4))
    print 'hstack1 : \n', np.hstack((a,b))

    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    print 'hstack2 : \n', np.hstack((a, b))

    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    print 'vstack1 : \n', np.vstack((a, b))

    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    print 'vstack2 : \n', np.vstack((a, b))

    a = np.array((1, 2, 3))
    b = np.array((2, 3, 4))
    print 'dstack1 : \n', np.dstack((a,b))

    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    print 'dstack2 : \n', np.dstack((a, b))

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    print 'concatenate1 : \n', np.concatenate((a, b), axis=0)
    print 'concatenate2 : \n', np.concatenate((a, b.T), axis=1)

    a = np.ma.arange(3)
    a[1] = np.ma.masked
    b = np.arange(2, 5)
    print 'a : \n', a
    print 'b : \n', b

   # print np.concatenate([a, b])
    x = np.arange(8.0).reshape(2, 2, 2)
    print 'x : \n', x
    print 'split : \n', np.vsplit(x, 2)
