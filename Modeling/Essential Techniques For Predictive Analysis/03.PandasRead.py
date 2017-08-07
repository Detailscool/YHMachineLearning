import pandas as pd

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = pd.read_csv(target_url, header=None, prefix='V')

print 'head : \n', data.head()
print 'tail : \n', data.tail()

summary = data.describe()
print 'summary : \n', summary

