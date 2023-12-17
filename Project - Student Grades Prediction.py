#!/usr/bin/env python
# coding: utf-8

# In[19]:


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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[20]:


df = pd.read_csv('Grades.csv')
df


# In[21]:


df.head()


# In[22]:


df.shape


# In[25]:


df.info()


# In[ ]:


# Observation :  All the features are in Object datatype, coverting same to intger.

df['PH-121'].replace('no', 0,inplace=True)
df['subscribed'].replace('yes', 1,inplace=True)
df


 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Seat No.   571 non-null    object 
 1   PH-121     571 non-null    object 
 2   HS-101     571 non-null    object 
 3   CY-105     570 non-null    object 
 4   HS-105/12  570 non-null    object 
 5   MT-111     569 non-null    object 
 6   CS-105     571 non-null    object 
 7   CS-106     569 non-null    object 
 8   EL-102     569 non-null    object 
 9   EE-119     569 non-null    object 
 10  ME-107     569 non-null    object 
 11  CS-107     569 non-null    object 
 12  HS-205/20  566 non-null    object 
 13  MT-222     566 non-null    object 
 14  EE-222     564 non-null    object 
 15  MT-224     564 non-null    object 
 16  CS-210     564 non-null    object 
 17  CS-211     566 non-null    object 
 18  CS-203     566 non-null    object 
 19  CS-214     565 non-null    object 
 20  EE-217     565 non-null    object 
 21  CS-212     565 non-null    object 
 22  CS-215     565 non-null    object 
 23  MT-331     562 non-null    object 
 24  EF-303     561 non-null    object 
 25  HS-304     561 non-null    object 
 26  CS-301     561 non-null    object 
 27  CS-302     561 non-null    object 
 28  TC-383     561 non-null    object 
 29  MT-442     561 non-null    object 
 30  EL-332     562 non-null    object 
 31  CS-318     562 non-null    object 
 32  CS-306     562 non-null    object 
 33  CS-312     561 non-null    object 
 34  CS-317     559 non-null    object 
 35  CS-403     559 non-null    object 
 36  CS-421     559 non-null    object 
 37  CS-406     486 non-null    object 
 38  CS-414     558 non-null    object 
 39  CS-419     558 non-null    object 
 40  CS-423     557 non-null    object 
 41  CS-412     492 non-null    object 
 42  CGPA       571 non-null    float64


# In[24]:


df.isna().sum()


# In[ ]:





# In[ ]:





# In[15]:


# Observation :  Most of the features have null valus. Maximum null values are in CS-406(85), CS-412(79) which is around 14.89% 
#                & 13.84% respectively. Rest of the features carrying null values have 2.63% null value contribution.

df.iplot()


# In[ ]:




