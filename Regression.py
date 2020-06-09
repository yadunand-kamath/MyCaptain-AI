# REGRESSION PROJECT

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.DESCR) 

dataset = boston.data
for name, index in enumerate(boston.feature_names):
    print(index, name)

# reshaping data
data = dataset[:,12].reshape(-1,1)
np.shape(dataset)

target = boston.target.reshape(-1,1)
np.shape(target)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'blue')
plt.xlabel("Lower Income Population")
plt.ylabel("Cost of house")
plt.show()

# regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(data, target)

# prediction
pred = reg.predict(data)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'red')
plt.plot(data, pred, color = 'green')
plt.xlabel("Lower Income Population")
plt.ylabel("Cost of house")
plt.show()

# polynomial model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(3), reg)
model.fit(data,target)

pred = model.predict(data)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'red')
plt.plot(data, pred, color = 'green')
plt.xlabel("Lower Income Population")
plt.ylabel("Cost of house")
plt.show()

# r-square metric
from sklearn.metrics import r2_score
r2_score(pred,target)
