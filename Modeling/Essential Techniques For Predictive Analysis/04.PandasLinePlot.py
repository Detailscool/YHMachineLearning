import pandas as pd
import matplotlib.pyplot as plt

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = pd.read_csv(target_url, header=None, prefix='V')

for i in range(208):
    if data.iat[i, 60] == 'M':
        pcolor = 'red'
    else:
        pcolor = 'blue'
    dataRow = data.iloc[i, 0:60]
    # print type(dataRow)
    dataRow.plot(color=pcolor)

plt.xlabel('Attribute Index')
plt.ylabel('Attribute Values')
plt.show()