# MACHINE LEARNING PROJECT

#!/usr/bin/env python
# coding: utf-8

# In[5]:
import pandas 
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier 


# In[7]:
# loading the data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = read_csv(url,names=names)


# In[9]:
# dataset
print(dataset.shape)
print(dataset.head(20))


# In[10]:
#statistical summary
print(dataset.describe())


# In[12]:
print(dataset.groupby('class').size())


# In[13]:
# plots
dataset.plot(kind = 'box',subplots = True,layout = (2,2), sharex = False, sharey = False)
pyplot.show()


# In[14]:
#histogram
dataset.hist()
pyplot.show()


# In[15]:
# multivariate plots
scatter_matrix(dataset)
pyplot.show()


# In[16]:
#splitting dataset
array  = dataset.values
x = array[:, 0:4]
y = array[:, 4]
X_train,X_validation,Y_train,Y_validation = train_test_split(x , y , test_size = 0.2 , random_state = 1)


# In[23]:
models = []
models.append(('LR' , LogisticRegression(solver = 'liblinear' , multi_class = 'ovr')))
models.append(('LDA' , LinearDiscriminantAnalysis()))
models.append(('KNN' , KNeighborsClassifier()))
models.append(('NB' , GaussianNB()))
models.append(('SVM' , SVC(gamma = 'auto')))

results = []
names = []
for name,model in models:
    kfold = StratifiedKFold(n_splits = 10 , random_state = 1)
    cv_results = cross_val_score(model , X_train , Y_train , cv = kfold , scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name , cv_results.mean() , cv_results.std()))


# In[24]:
# compare models
pyplot.boxplot(results , labels = names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[27]:
# predictions on SVM
model = SVC(gamma = 'auto')
model.fit(X_train , Y_train)
predictions = model.predict(X_validation)


# In[30]:
# evaluate predictions
print(accuracy_score(Y_validation , predictions))
print(confusion_matrix(Y_validation , predictions))
print(classification_report(Y_validation , predictions))

