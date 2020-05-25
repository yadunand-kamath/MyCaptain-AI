#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[3]:


boston = load_boston()
print(boston.DESCR) 


# In[4]:


dataset = boston.data
for name, index in enumerate(boston.feature_names):
    print(index, name)


# In[6]:


# reshaping data
data = dataset[:,12].reshape(-1,1)
np.shape(dataset)


# In[7]:


target = boston.target.reshape(-1,1)
np.shape(target)


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'blue')
plt.xlabel("Lower Income Population")
plt.ylabel("Cost of house")
plt.show()


# In[10]:


# regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(data, target)


# In[11]:


# prediction
pred = reg.predict(data)


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'red')
plt.plot(data, pred, color = 'green')
plt.xlabel("Lower Income Population")
plt.ylabel("Cost of house")
plt.show()


# In[14]:


# polynomial model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[15]:


model = make_pipeline(PolynomialFeatures(3), reg)
model.fit(data,target)


# In[16]:


pred = model.predict(data)


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'red')
plt.plot(data, pred, color = 'green')
plt.xlabel("Lower Income Population")
plt.ylabel("Cost of house")
plt.show()


# In[18]:


# r-square metric
from sklearn.metrics import r2_score
r2_score(pred,target)


# In[ ]:




