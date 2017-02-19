import numpy as np

if __name__ == "__main__":
    arr = np.random.rand(5, 4)
    print 'arr:\n', arr

    mean = arr.mean(axis=1)
    print 'mean: ', mean

    sum = arr.sum(0)
    print 'sum: ', sum

    arr = np.stack(np.tile([1, 2, 3], [3, 1]),axis=1)
    # arr = np.linspace(1, 10, 10)
    print 'arr:\n', arr

    cumsum = arr.cumsum()
    print 'cumsum: \n', cumsum

    cumprod = arr.cumprod()
    print 'cumprod:\n ',cumprod
