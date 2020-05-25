#!/usr/bin/env python
# coding: utf-8

# In[2]:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:
df_train = pd.read_csv('mnist_train.csv')
df_test = pd.read_csv('mnist_test.csv')


# In[4]:
df_train.head()


# In[5]:
a = df_train.iloc[2,1:].values


# In[6]:
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[7]:
df_x = df_train.iloc[:,1:]
df_y = df_train.iloc[:,0]


# In[8]:
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)


# In[9]:
y_train.head()


# In[10]:
rf = RandomForestClassifier(n_estimators=100)


# In[11]:
rf.fit(x_train, y_train)


# In[12]:
pred = rf.predict(x_test)
pred


# In[14]:
s = y_test.values
count = 0
for i in range(len(pred)):
    if pred[i]  == s[i]:
        count = count+1


# In[15]:
# number of predicted values 
count


# In[16]:
# total values
len(pred)


# In[17]:
# accuracy
11620/12000

