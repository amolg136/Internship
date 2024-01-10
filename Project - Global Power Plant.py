#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


power_plant= pd.read_csv('database_IND.csv')
power_plant


# In[3]:


power_plant.shape


# In[4]:


power_plant.info()


# In[5]:


power_plant.isna().sum()


# In[6]:


power_plant.columns


# In[7]:


power_plant["country"].unique()


# In[8]:


power_plant["country_long"].unique()


# In[9]:


power_plant["primary_fuel"].unique()


# In[10]:


power_plant["other_fuel1"].unique()


# In[11]:


power_plant["other_fuel2"].unique()

# Observation :- Database is for the Single country "India", hence removing the "country" & "country_long" column.
# Observation :- Most of the features such as 'country', 'country_long', 'name', 'gppd_idnr', 'latitude', 'longitude',                         'primary_fuel', 'other_fuel1', 'other_fuel2', 'other_fuel3', 'commissioning_year', 'owner', 'source', 'url',
                'geolocation_source', 'wepp_id', 'year_of_capacity_data', 'generation_gwh_2013', 'generation_gwh_2019',                         'generation_data_source', 'estimated_generation_gwh' has "null" values and are not the continuous values hence                 removing the same. 
# In[12]:


power_plant.drop(['country', 'country_long', 'name', 'gppd_idnr', 'latitude', 'longitude', 'other_fuel3', 
                  'commissioning_year', 'owner', 'source', 'url', 'geolocation_source', 'wepp_id', 
                  'year_of_capacity_data', 'generation_gwh_2013', 'generation_gwh_2019', 'generation_data_source', 
                  'estimated_generation_gwh'], axis=1, inplace=True)
power_plant


# In[13]:


power_plant.columns


# In[14]:


power_plant.info()


# In[15]:


power_plant.isna().sum()


# In[16]:


power_plant.plot(kind = 'bar', figsize = (8, 6))
plt.xlabel('capacity_mw', color = 'g', fontsize = 15)
plt.ylabel('generation_gwh_2014', color = 'r', fontsize = 15)
plt.grid(axis = 'y')
plt.yticks(range(0, 25, 2))
plt.show()

