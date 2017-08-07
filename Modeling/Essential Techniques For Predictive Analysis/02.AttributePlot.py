import urllib2
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = urllib2.urlopen(target_url)
print type(data)

xList = []
labels = []

for line in data:
    row = line.strip().split(",")
    xList.append(row)

data = np.array(xList)
x, y = np.split(data, (60, ), axis=1)
x = np.array(x, dtype=float)

# Test
x = x[:, 3]

stats.probplot(x.ravel(), dist='norm', plot=plt)
plt.show()