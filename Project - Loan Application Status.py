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
from scipy.stats import zscore
import pickle
from sklearn import metrics
from sklearn import preprocessing
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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 


# In[2]:


loan_app_status = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv')
loan_app_status.head()


# In[3]:


loan_app_status.shape


# In[4]:


loan_app_status.info()


# In[5]:


loan_app_status.dtypes.value_counts()


# In[6]:


loan_app_status.isna().sum()


# In[7]:


loan_app_status.columns


# In[8]:


# Dropping "Loan_ID" Column as it carries unique values like index.

loan_app_status.drop(['Loan_ID'],axis=1,inplace=True)
loan_app_status


# In[9]:


# Visualize all the unique values in columns using barplot.

obj = (loan_app_status.dtypes == 'object') 
object_cols = list(obj[obj].index) 
plt.figure(figsize=(18,36)) 
index = 1
  
for col in object_cols: 
  y = loan_app_status[col].value_counts() 
  plt.subplot(11,4,index) 
  plt.xticks(rotation=90) 
  sns.barplot(x=list(y.index), y=y) 
  index +=1


# In[10]:


# All the categorical values are binary, hence using Label Encoder for all columns to change values into int datatype.

label_encoder = preprocessing.LabelEncoder() 
obj = (loan_app_status.dtypes == 'object') 
for col in list(obj[obj].index): 
  loan_app_status[col] = label_encoder.fit_transform(loan_app_status[col])


# In[11]:


# Checking the object datatype columns.

obj = (loan_app_status.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))


# In[12]:


# Plotting a Heatmap.

plt.figure(figsize=(12,6)) 
  
sns.heatmap(loan_app_status.corr(),cmap='BrBG',fmt='.2f', linewidths=2,annot=True)


# There is correlation between Loan Amount and ApplicantIncome. Also Credit_History has a high impact on Loan_Status.

# In[13]:


# Using Catplot to visualize the plot for the Gender, and Marital Status of the applicant.

sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=loan_app_status)


# In[14]:


for col in loan_app_status.columns: 
  loan_app_status[col] = loan_app_status[col].fillna(loan_app_status[col].mean())

loan_app_status.isna().sum()


# In[15]:


# Splitting Datases

X = loan_app_status.drop(['Loan_Status'],axis=1) 
Y = loan_app_status['Loan_Status'] 
X.shape,Y.shape 
  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[16]:


# Using KNN Model

knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators = 7, criterion = 'entropy', random_state =7) 
svc = SVC() 
lc = LogisticRegression() 
  
# making predictions on the training set 
for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_train) 
    print("Accuracy score of ", clf.__class__.__name__, "=",100*metrics.accuracy_score(Y_train, Y_pred))    


# In[17]:


# Making predictions on the testing set

for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test) 
    print("Accuracy score of ", clf.__class__.__name__,"=", 100*metrics.accuracy_score(Y_test, Y_pred))


# In[ ]:




