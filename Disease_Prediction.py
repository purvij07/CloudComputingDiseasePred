#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[35]:


train = pd.read_csv("C:/Users/purvi/Downloads/CloudComputingProj/Training.csv")
test = pd.read_csv('C:/Users/purvi/Downloads/CloudComputingProj/Testing.csv')
A = train
B = test


# In[36]:


A.head()


# In[37]:


B.head()


# In[38]:


A = A.drop(["Unnamed: 133"],axis=1)


# In[39]:


A.prognosis.value_counts()


# In[40]:


A.isna().sum()


# In[41]:


B.isna().sum()


# In[42]:


Y = A[["prognosis"]]
X = A.drop(["prognosis"],axis=1)
P = B.drop(["prognosis"],axis=1)


# In[43]:


xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[44]:


dtc= DecisionTreeClassifier(random_state=42)
model_dtc = dtc.fit(xtrain,ytrain)
tr_pred_dtc = model_dtc.predict(xtrain)
ts_pred_dtc = model_dtc.predict(xtest)

print("training accuracy is:",accuracy_score(ytrain,tr_pred_dtc))
print("testing accuracy is:",accuracy_score(ytest,ts_pred_dtc))


# In[48]:


random_forest_classifier = RandomForestClassifier(random_state=0)
random_forest_classifier.fit(xtrain,ytrain)
ypred=random_forest_classifier.predict(xtest)


# In[49]:


print("Accuracy:",round(metrics.accuracy_score(ytest, ypred)* 100,2),"%")


# In[53]:


decisionTree = DecisionTreeClassifier(random_state=0)


# In[54]:


decisionTree.fit(xtrain,ytrain)


# In[57]:


ypred=decisionTree.predict(xtest)


# In[58]:


print("Accuracy:",round(metrics.accuracy_score(ytest, ypred)* 100,2),"%")


# In[60]:


from sklearn import svm
SVM = svm.SVC(decision_function_shape='ovo',kernel='poly')
SVM.fit(xtrain,ytrain)
ypred = SVM.predict(xtest)


# In[61]:


print("Accuracy:",round(metrics.accuracy_score(ytest, ypred)* 100,2),"%")


# In[62]:
import pickle 
pickle.dump(random_forest_classifier,open('model.pkl','wb'))

#test.join(pd.DataFrame(model_dtc.predict(P),columns=["predicted"]))[["prognosis","predicted"]]


# In[ ]:
input_features = np.zeros(132)
input_features[[0, 4, 9]] = [4, 56, 10]
prediction = random_forest_classifier.predict(input_features.reshape(1, -1))




