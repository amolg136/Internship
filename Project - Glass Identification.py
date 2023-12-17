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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


df = pd.read_csv('glass.csv')
df


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


# Observations 1) Row containing data is used as header. Adding header to data & creating a new dataframe (df1).

df1 = pd.read_csv('glass.csv', names=['Sr.','RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type of Glass'])
df1


# In[6]:


df1.shape


# In[7]:


df1.info()


# In[8]:


df1.isna().sum()


# In[9]:


# Observations 2) The dataset does not have any null values.
# Observations 3) Removing Serial number column (Sr.) as Indextation of data is already there.

df1.drop('Sr.', axis=1, inplace=True)
df1


# In[10]:


df1.info()


# In[11]:


df1.describe()


# In[12]:


# Observations 4) Data is not normally distributed. The features Potassium(K), Calcium(Ca), Barium(Ba), Iron(Fe).
# Observations 5) The distribution of Potassium(K) and Barium(Ba) have many outliers. Validating tha same with ditribution plot.

plt.figure(figsize=(20,15), facecolor = "orange")
plotnumber = 1

for column in df1:
    if plotnumber<=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(df1[column])
        plt.xlabel(column, fontsize = 20)
    plotnumber+=1
plt.tight_layout()


# In[13]:


df1['Type of Glass'].value_counts()


# In[14]:


# Observations 6) Glass Type 2 & Type 1 are have highest occurances in the dataset. We'll plot the same in countplot.

sns.countplot(x = 'Type of Glass', data = df1)
plt.show()


# In[15]:


# Observations 7) Data distribution seems normal in all columns except Iron (Fe).
# Plot the box plot to look at the distribution of the different features of this dataset.

sns.set(style="whitegrid", font_scale=1.2)
plt.subplots(figsize = (20,15))
plt.subplot(3,3,1)
sns.boxplot(x='Type of Glass', y='RI', data=df1)
plt.subplot(3,3,2)
sns.boxplot(x='Type of Glass', y='Na', data=df1)
plt.subplot(3,3,3)
sns.boxplot(x='Type of Glass', y='Mg', data=df1)
plt.subplot(3,3,4)
sns.boxplot(x='Type of Glass', y='Al', data=df1)
plt.subplot(3,3,5)
sns.boxplot(x='Type of Glass', y='Si', data=df1)
plt.subplot(3,3,6)
sns.boxplot(x='Type of Glass', y='K', data=df1)
plt.subplot(3,3,7)
sns.boxplot(x='Type of Glass', y='Ca', data=df1)
plt.subplot(3,3,8)
sns.boxplot(x='Type of Glass', y='Ba', data=df1)
plt.subplot(3,3,9)
sns.boxplot(x='Type of Glass', y='Fe', data=df1)
plt.show()


# In[16]:


# Observations 8) Glasss Type 1, Type 2, Type 4, Type 5 & Type 7 have Na, Mg, Al, Si, K, Ca, Ba & Iron Content.
# Observations 9) Magnesium (Mg) content is higher in Type 5 ^ Type 6 Glass.
# Observations 10) Barium (Ba) content is high in Type 7 Glass, other Glass Type has negligible Iron (Fe) Content.
# Observations 11) Glass Type 5 has higer Mean and wider range of all the components.
# Observations 12) Iron Component is absent in Glass Typoe 5 and mimimum present in Glass Type 5 & Type 7.
# Observations 13) the distribution of potassium (K) and Barium (Ba) seem to contain many outliers.

plt.subplots(figsize=(15,10))
sns.heatmap(df1.corr(),cmap='YlGnBu',annot=True, linewidth=.5)


# In[17]:


# Observations 14) Refractive Index (RI) & Calcium (Ca) has high positive corelation wheareas RI & Si have negative corelation.

df1.skew()


# In[18]:


df.corr()


# In[19]:


x = df1.drop(columns = ["Type of Glass"])
y = df1['Type of Glass']


# In[20]:


x


# In[21]:


y


# In[22]:


# Data Standardization

scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)


# In[23]:


x_scaled


# In[24]:


x_scaled.shape[1]


# In[25]:


x_scaled


# In[26]:


# Applying Variance Inflation Factor (variance_inflation_factor)

vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
vif['Features'] = x.columns
vif


# In[27]:


# Splitting data into Train & Text Datasets

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.25 , random_state = 130)
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred


# In[29]:


accuracy = r2_score(y_test, y_pred)
accuracy


# In[ ]:




