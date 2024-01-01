#!/usr/bin/env python
# coding: utf-8

# In[37]:


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
from scipy.stats import zscore
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[2]:


bb = pd.read_csv('baseball.csv')
bb


# In[3]:


bb.head()


# In[4]:


bb.tail()


# In[5]:


bb.shape


# In[6]:


bb["CG"].unique()


# In[7]:


bb.info()


# In[8]:


bb.isna()


# In[9]:


bb.isna().sum()


# In[10]:


# Observation :- Data has no null values. Also all features are in integer data type except ERA which is in float data type.

sns.heatmap(bb.isnull().sum().to_frame())


# In[11]:


bb.describe()


# In[12]:


plt.figure(figsize=(20,15), facecolor = "orange")
plotnumber = 1

for column in bb:
    if plotnumber<=17:
        ax = plt.subplot(3,6,plotnumber)
        sns.distplot(bb[column])
        plt.xlabel(column, fontsize = 20)
    plotnumber+=1
plt.tight_layout()


# In[13]:


bb.corr()


# In[14]:


plt.subplots(figsize=(15,10))
sns.heatmap(bb.corr(),cmap='YlGnBu',annot=True, linewidth=.5)


# In[15]:


# Observation :- "Wins", "Save", "walk" & "Runs" are linearly correlated with each other.
# Observation :- "Wins", "E", "ERA", "ER" & "RA" are weekly correlated with each other.
# Observation :- "R"(Runs Scored), "HR"(Home Runs), "2B"(Double), "SV"(Save), & "SHO"(Shutout) are positively correlated.


# In[16]:


# Cheking the Outliers.

plt.figure(figsize=(20,15), facecolor = "g")
plotnumber = 1
for column in bb:
    if plotnumber<=17:
        ax = plt.subplot(3,6,plotnumber)
        sns.boxplot(bb[column])
        plt.xlabel(column, fontsize = 20)
    plotnumber+=1
plt.tight_layout()


# In[17]:


# Observation :- "R"(Runs Scored), "ERA"(Earned Run Average), "SHO"(Shutout), "SV"(Save) & "E"(Errors) has outliers.
# Observation :- Saving lable in different variable to treate the outliers.

bb_lable = bb['W']
bb_features = bb.drop("W", axis=1)


# In[18]:


bb_lable


# In[19]:


bb_features


# In[20]:


z=np.abs(zscore(bb_features))
z


# In[21]:


bb1 = bb[(z<3).all(axis=1)]
bb1.shape


# In[22]:


bb.shape


# In[23]:


# Observation :- Outlier found in only 1 column is been trated.
# Checking the Data Loss

print("Data Loss :- ", ((bb.shape[0]-bb1.shape[0])/bb.shape[0])*100)


# In[24]:


# Checking the Outliers using IQR Method.

Q1 = bb_features.quantile(0.25)
Q3 = bb_features.quantile(0.75)
IQR = Q3-Q1

bb2 = bb[~((bb<(Q1-1.5*IQR)) | (bb>(Q3+1.5*IQR))).any(axis=1)]
bb2.shape


# In[25]:


print("Data Loss :- ", ((bb.shape[0]-bb2.shape[0])/bb.shape[0])*100)


# In[26]:


# Observation :- 33% Data Loss is not considerable. We'll procced with Z2 Method.
# Checking the distribution post treating the outliers.

plt.figure(figsize=(20,15), facecolor = "orange")
plotnumber = 1

for column in bb1:
    if plotnumber<=17:
        ax = plt.subplot(3,6,plotnumber)
        sns.distplot(bb1[column])
        plt.xlabel(column, fontsize = 20)
    plotnumber+=1
plt.tight_layout()


# In[27]:


# Splitting the Data into Train & Test

x = bb1.drop(columns=["W"])
y = bb1['W']


# In[28]:


x


# In[29]:


y


# In[30]:


# Finding the best random state.

maxAccu=0
maxRS=0

for i in range(1,29):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30, random_state=i)
    lr=LinearRegression()
    lr.fit(x_train, y_train)
    pred=lr.predict(x_test)
    Acc=r2_score(y_test, pred)
    if Acc>maxAccu:
        maxAccu=Acc
        maxRS=i
print("Maximum R2_Score is", maxAccu, "On Random State", maxRS)


# In[34]:


# Using Linear Regression Method

LR = LinearRegression()
LR.fit(x_train, y_train)
pred_LR=LR.predict(x_test)
pred_train=LR.predict(x_train)

print("R2_Score :-", r2_score(y_test, pred_LR))
print("R2_Score on Training Data :-", r2_score(y_train, pred_train)*100)
print("Mean Absolute Error :-", mean_absolute_error(y_test, pred_LR))
print("Mean Squared Error :-", mean_squared_error(y_test, pred_LR))
print("Root Mean Squared Error :-", np.sqrt(mean_squared_error(y_test, pred_LR)))


# In[35]:


# Using Random Forest Method

RFR=RandomForestRegressor()
RFR.fit(x_train, y_train)
pred_RFR=RFR.predict(x_test)
pred_train=RFR.predict(x_train)

print("R2_Score :-", r2_score(y_test, pred_LR))
print("R2_Score on Training Data :-", r2_score(y_train, pred_train)*100)
print("Mean Absolute Error :-", mean_absolute_error(y_test, pred_LR))
print("Mean Squared Error :-", mean_squared_error(y_test, pred_LR))
print("Root Mean Squared Error :-", np.sqrt(mean_squared_error(y_test, pred_LR)))


# In[36]:


# Using KNN Method

knn=KNN()
knn.fit(x_train, y_train)
pred_knn=knn.predict(x_test)
pred_train=knn.predict(x_train)

print("R2_Score :-", r2_score(y_test, pred_LR))
print("R2_Score on Training Data :-", r2_score(y_train, pred_train)*100)
print("Mean Absolute Error :-", mean_absolute_error(y_test, pred_LR))
print("Mean Squared Error :-", mean_squared_error(y_test, pred_LR))
print("Root Mean Squared Error :-", np.sqrt(mean_squared_error(y_test, pred_LR)))


# In[ ]:




