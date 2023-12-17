#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from numpy.random import randn
from pandas.plotting import scatter_matrix
import scipy.stats as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[3]:


df_train= pd.read_csv('termdeposit_train.csv')
df_train


# In[4]:


df_test= pd.read_csv('termdeposit_test.csv')
df_test


# In[6]:


df_train.shape


# In[7]:


df_test.shape


# In[8]:


df_train.info()


# In[9]:


df_test.info()


# In[10]:


df_train.isna().sum()


# In[11]:


df_test.isna().sum()


# In[12]:


# Observations : 1. There are no null Values in both train & test data.

df_train.describe()


# In[13]:


df_test.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


# Plotting the 'subscribed' frequency
sns.countplot(data=df_train, x='subscribed')


# In[20]:


df_train['subscribed'].value_counts()


# In[22]:


# Observations : 2. 3715 people out of 31647 have subscribed which is roughly 12%.

df_train['job'].value_counts()


# In[24]:


sns.set_context('paper')
df_train['job'].value_counts().plot(kind='bar', figsize=(10,6));


# In[28]:


# Observations : 3. Maximum clients belongs to blue-collared, management, technician, admin & services job.
#                   Unemployed Housemaid & student are in minimum numbers.

sns.distplot(df_train['age'])
plt.xlabel(['age'], fontsize = 20)


# In[29]:


print(pd.crosstab(df_train['job'],df_train['subscribed']))


# In[33]:


job = pd.crosstab(df_train['job'],df_train['subscribed'])
job_norm = job.div(job.sum(1).astype(float), axis=0)
job_norm.plot.bar(stacked=True,figsize=(8,6));


# In[35]:


# Converting the target variables into 0s and 1s

df_train['subscribed'].replace('no', 0,inplace=True)
df_train['subscribed'].replace('yes', 1,inplace=True)
df_train


# In[37]:


#Correlation matrix
tc = df_train.corr()
tc


# In[39]:


fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')


# In[41]:


# Observations : 2. We can infer that duration of the call is highly correlated with the target variable. As the duration of 
#                   the call is more, there are higher chances that the client is showing interest in the term deposit and 
#                   hence there are higher chances that the client will subscribe to term deposit.

target = df_train['subscribed']
df_train = df_train.drop('subscribed', axis=1)


# In[45]:


df_train = pd.get_dummies(df_train)
df_train.head()


# In[47]:


X_train, X_val, y_train, y_val = train_test_split(df_train, target, test_size=0.2, random_state=136)


# In[50]:


lr = LogisticRegression()


# In[52]:


lr.fit(X_train,y_train)


# In[53]:


pred = lr.predict(X_val)


# In[54]:


accuracy_score(y_val,pred)


# In[57]:


clf = DecisionTreeClassifier(max_depth=4, random_state=0)


# In[58]:


clf.fit(X_train, y_train)


# In[59]:


predict = clf.predict(X_val)
predict


# In[60]:


accuracy_score(y_val,predict)


# In[62]:


df_test = pd.get_dummies(df_test)
df_test.head()


# In[64]:


df_test_pred = clf.predict(df_test)
df_test_pred


# In[66]:


Bank_Marketing = pd.DataFrame()


# In[69]:


Bank_Marketing['ID'] = df_test['ID']
Bank_Marketing['subscribed'] = df_test_pred


# In[70]:


Bank_Marketing['subscribed']


# In[71]:


Bank_Marketing['subscribed'].replace(0,'no',inplace=True)
Bank_Marketing['subscribed'].replace(1,'yes',inplace=True)


# In[72]:


Bank_Marketing['subscribed']


# In[73]:


Bank_Marketing.to_csv('submission file.csv', header=True, index=False)


# In[ ]:




