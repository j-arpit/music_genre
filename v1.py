#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')


# In[3]:


train


# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


# there are some cols which contains '?' instead nan
col_list = []
for i in train.columns:
    l = train[i].unique()
    if '?' in l:
        col_list.append(i)
print(col_list)
train["mode"] = np.where(train["mode"] == "?", np.nan, train["mode"])
train["tempo"] = np.where(train["tempo"] == "?", np.nan, train["tempo"])
train['tempo'] = train['tempo'].astype('float64')


# In[7]:


test["mode"] = np.where(test["mode"] == "?", np.nan, test["mode"])
test["tempo"] = np.where(test["tempo"] == "?", np.nan, test["tempo"])
test['tempo'] = test['tempo'].astype('float64')


# In[8]:


train_cols_with_missing_values = [col for col in train.columns if train[col].isnull().sum()>0]
# out of these voice_gender, mode, musician_category are object datatype so
obj_list = ['voice_gender', 'mode', 'musician_category']
train_cols_with_missing_values = list(set(train_cols_with_missing_values) - set(obj_list))
train_cols_with_missing_values


# In[9]:


test_cols_with_missing_values = [col for col in test.columns if test[col].isnull().sum()>0]
# out of these voice_gender, mode, musician_category are object datatype so
test_cols_with_missing_values = list(set(test_cols_with_missing_values) - set(obj_list))
test_cols_with_missing_values


# In[10]:


for col in obj_list:
    train[col].fillna(value=train[col].mode()[0], inplace=True)
    test[col].fillna(value=test[col].mode()[0], inplace=True)
for col in train_cols_with_missing_values:
    train[col].fillna(value=train[col].mean(), inplace=True)
for col in test_cols_with_missing_values:
    test[col].fillna(value=test[col].mean(), inplace=True)


# In[11]:


print("Train cols which contains missing data ", [col for col in train.columns if train[col].isnull().sum()>0])
print("test cols which contains missing data ", [col for col in test.columns if test[col].isnull().sum()>0])


# In[12]:


obj_list.append("key")
train = pd.get_dummies(train, columns = obj_list)
test = pd.get_dummies(test, columns = obj_list)


# In[13]:


drop_list = ['instance_id', 'track_name', 'music_genre']
X = train.drop(drop_list, axis =1)
y = train['music_genre'].copy()


# In[14]:


from sklearn.model_selection import train_test_split as tsp


# In[15]:


X_train, X_val, y_train, y_val = tsp(X, y, test_size = 0.2)


# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)


# In[17]:


from sklearn.svm import SVC
import time
clf = SVC()
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
print(end - start)
y_pred = clf.predict(X_val)
from sklearn import metrics
score = 100 * (metrics.f1_score(y_val, y_pred, average="macro"))
score


# In[18]:


clf = SVC(kernel='poly')
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
print(end - start)
y_pred = clf.predict(X_val)
from sklearn import metrics
score = 100 * (metrics.f1_score(y_val, y_pred, average="macro"))
score


# In[19]:


clf = SVC(kernel='linear', decision_function_shape='ovo')
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
print(end - start)
y_pred = clf.predict(X_val)
from sklearn import metrics
score = 100 * (metrics.f1_score(y_val, y_pred, average="macro"))
score


# In[20]:


# final classifier
clf = SVC(kernel='linear', C = 10)
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
print(end - start)
y_pred = clf.predict(X_val)
from sklearn import metrics
score = 100 * (metrics.f1_score(y_val, y_pred, average="macro"))
score


# In[21]:


test


# In[22]:


submission = test['instance_id'].copy()


# In[23]:


X_test = test.drop(['instance_id', 'track_name'], axis =1)


# In[24]:


X_test


# In[25]:


scaler_test = StandardScaler()
scaler_test.fit(X_test)
X_test = scaler_test.transform(X_test)


# In[26]:


y_test_pred = y_pred = clf.predict(X_test)


# In[27]:


y_new = pd.Series(y_test_pred)


# In[30]:


submission = pd.concat([submission, y_new], axis=1)


# In[31]:


submission


# In[32]:


submission.to_csv("submission.csv", index=False)


# In[ ]:


parameters = {'kernel': ['linear', 'poly', 'rbf'], 
              'C':[0.001, 0.1, 100, 10000], 
              'gamma':[10,1,0.1,0.01], 
              'decision_function_shape':['ovo', 'ovr']}
from sklearn.model_selection import GridSearchCV as gs
grid = gs(SVC(), param_grid=parameters, cv =2)
grid.fit(X_val, y_val)
print("Score= {}").format(grid.score(X_test, y_test))


# In[ ]:




