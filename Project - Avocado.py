#!/usr/bin/env python
# coding: utf-8

# In[98]:


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
from sklearn.preprocessing import LabelEncoder
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


# In[99]:


avocado = pd.read_csv('avocado.csv')
avocado


# In[100]:


avocado.head()


# In[101]:


avocado.shape


# In[102]:


avocado.info()


# In[103]:


avocado.isna().sum()


# In[104]:


# Observation:- Removing null values.

avocado = avocado.dropna(how = 'all')
avocado


# In[105]:


avocado.shape


# In[106]:


# Observation:- Column "Unnamed: 0" refer to index values, dropping the same.

avocado.drop("Unnamed: 0", axis=1,inplace=True)
avocado


# In[107]:


avocado.describe()


# In[108]:


# Observation:- The average price of avocado is 1.405 and minimum 0.44 maximum is 3.25

avocado.columns


# In[109]:


avocado.type.unique()


# In[111]:


avocado['type'].replace({'conventional':'1'}, inplace=True)
avocado.head()


# In[112]:


sns.heatmap(avocado.corr())


# In[113]:


le = LabelEncoder()
avocado['region'] = le.fit_transform(avocado['region'])
avocado.head()


# In[114]:


avocado.region.unique()


# In[115]:


sns.heatmap(avocado.corr())


# In[116]:


sns.distplot(avocado["AveragePrice"],axlabel="Distribution of average price")


# In[117]:


# Observation:- Most Average price lies range from 1.0 to 1.7

sns.boxplot(x="year", y="AveragePrice", data=avocado)


# In[118]:


# Observation:- The average price was high in 2017 compared to other years.

avocado.groupby("region")["AveragePrice"].sum().sort_values(ascending=False).plot(kind="bar",figsize=(15,5))


# In[122]:


#Creating a new dataframe with few columns only and create train and test data.
avocado_new=avocado[["AveragePrice","Total Volume","region","type","Total Bags","year"]]

X=avocado_new[["AveragePrice","Total Volume","region","Total Bags","year"]] #feature columns
y=avocado_new.type #predictor variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print("X Train Shape ",X_train.shape)
print("Y Train Shape ",y_train.shape)

print("X Test Shape ",X_test.shape)
print("Y Test Shape ",y_test.shape)

scaler = StandardScaler().fit(avocado_new)
avocado_new_std = scaler.transform(avocado_new)
avocado_new = pd.DataFrame(avocado_new_std)
avocado_new.columns = ["AveragePrice","Total Volume","region","type","Total Bags","year"]
avocado_new.head()


# In[123]:


feature_cols = ['Total Volume', "region",'type','Total Bags', 'year']
# create a Python list of feature names

X = avocado_new[feature_cols]
# use the list to select a subset of the original DataFrame-+
y = avocado_new.AveragePrice


def split(X,y):
    return train_test_split(X, y, test_size=0.20, random_state=1)

X_train, X_test, y_train, y_test=split(X,y)

print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)

print('X_test shape: ',X_test.shape)
print('y_test shape: ',y_test.shape)


# In[124]:


linreg1 = LinearRegression()
linreg1.fit(X_train, y_train) 

feature_cols.insert(0,'Intercept')
coef = linreg1.coef_.tolist()
coef.insert(0, linreg1.intercept_)
eq1 = zip(feature_cols, coef)

for c1,c2 in eq1:
    print(c1,c2)

y_pred_train = linreg1.predict(X_train)
y_pred_test = linreg1.predict(X_test)


# In[130]:


print("Model Evaluation for Linear Regression Model 1")

RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print('RMSE for training set is {}'.format(RMSE_train),' and RMSE for test set is {}'.format(RMSE_test))

yhat = linreg1.predict(X_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print("r_squared for test data ",r_squared, " and adjusted_r_squared for test data",adjusted_r_squared)

yhat = linreg1.predict(X_test)
SS_Residual = sum((y_test-yhat)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r_squared for test data ",r_squared, " and adjusted_r_squared for test data",adjusted_r_squared)


# In[ ]:




