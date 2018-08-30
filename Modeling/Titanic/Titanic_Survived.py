#!/usr/bin/python
# -*- coding:utf-8 -*-
#  Titanic_Survived.py
#  Created by HenryLee on 2018/8/5.
#  Copyright © 2018年. All rights reserved.
#  Description :


import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 18)

df = pd.read_csv("./14.Titanic.train.csv")
# print df.info()

df.loc[(df.Age.isnull()), 'Age'] = df.Age.dropna().mean()
print df.groupby('Survived').count()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df['Age'])
# df['Fare_scaled'] = scaler.fit_transform(df['Fare'])
#
# df.loc[df.Cabin.notnull(), 'Cabin'] = 'YES'
# df.loc[df.Cabin.isnull(), 'Cabin'] = 'NO'

df_pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
print df_pclass.head(3)
