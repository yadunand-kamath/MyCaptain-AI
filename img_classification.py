# IMAGE CLASSIFICATION PROJECT
# downloaded MNIST datasets from "https://www.kaggle.com/oddrationale/mnist-in-csv/data"

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

df_train = pd.read_csv('mnist_train.csv')
df_test = pd.read_csv('mnist_test.csv')

df_train.head()

a = df_train.iloc[2,1:].values

a = a.reshape(28,28).astype('uint8')
plt.imshow(a)

df_x = df_train.iloc[:,1:]
df_y = df_train.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)

y_train.head()

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

pred = rf.predict(x_test)
pred

s = y_test.values
count = 0
for i in range(len(pred)):
    if pred[i]  == s[i]:
        count = count+1

# number of predicted values 
count

# total values
len(pred)

# accuracy
11620/12000

