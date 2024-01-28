#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy.random import randn
from pandas.plotting import scatter_matrix
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import scipy.stats as st
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


# In[2]:


ins_fraud = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/Automobile_insurance_fraud.csv')
ins_fraud.head()


# In[3]:


ins_fraud.shape


# In[4]:


ins_fraud.info()


# In[5]:


ins_fraud.dtypes.value_counts()


# In[6]:


ins_fraud.nunique()


# In[7]:


ins_fraud.isna().sum()


# In[8]:


# heatmap  
  
plt.figure(figsize = (18, 12))
corr = ins_fraud.corr()
sns.heatmap(data = corr, annot = True, fmt = '.2g', linewidth = 1)
plt.show()


# In[9]:


# Below columns will not have any impact on analysis hence dropping the same from dataframe.  
  
drop_list = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year', '_c39']  
ins_fraud.drop(drop_list, inplace = True, axis = 1)
ins_fraud.head()


# In[10]:


# checking for multicollinearity  
  
plt.figure(figsize = (18, 12))  
corr = ins_fraud.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))  
sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
plt.show()


# From the above plot, we can see that there is a high correlation between age and months_as_customer. We will drop the "Age" column. Also, there is a high correlation between total_clam_amount, injury_claim, property_claim, and vehicle_claim, as the total claim is the sum of all others. So we will drop the total claim column.

# In[11]:


ins_fraud.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)
ins_fraud.head()


# In[12]:


ins_fraud.info()


# In[13]:


# separating the feature and target columns  
  
X = ins_fraud.drop('fraud_reported', axis = 1)  
y = ins_fraud['fraud_reported']  


# In[14]:


# extracting categorical columns  
ins_fraud_cat = X.select_dtypes(include = ['object'])  


# In[15]:


ins_fraud_cat.head()


# In[16]:


# printing unique values of each column  
for col in ins_fraud_cat.columns:  
    print(f"{col}: \n{ins_fraud_cat[col].unique()}\n")  


# In[17]:


ins_fraud_cat = pd.get_dummies(ins_fraud_cat, drop_first = True)  
ins_fraud_cat.head()


# In[18]:


# extracting the numerical columns  
  
ins_fraud_num = X.select_dtypes(include = ['int64'])  
ins_fraud_num.head()  


# In[19]:


# combining the Numerical and Categorical dataframes to get the final dataset  
  
X = pd.concat([ins_fraud_num, ins_fraud_cat], axis = 1)  
X.head()  


# In[20]:


plt.figure(figsize = (25, 20))  
plotnumber = 1  
  
for col in X.columns:  
    if plotnumber <= 24:  
        ax = plt.subplot(5, 5, plotnumber)  
        sns.distplot(X[col])  
        plt.xlabel(col, fontsize = 15)  
         
    plotnumber += 1  
     
plt.tight_layout()  
plt.show()  


# The data looks good. Let's check for outliers.

# In[21]:


plt.figure(figsize = (20, 15))  
plotnumber = 1  
  
for col in X.columns:  
    if plotnumber <= 24:  
        ax = plt.subplot(5, 5, plotnumber)  
        sns.boxplot(X[col])  
        plt.xlabel(col, fontsize = 15)  
     
    plotnumber += 1  
plt.tight_layout()  
plt.show()  


# Outliers are present in some numerical columns. We will scale numerical columns later.

# In[22]:


# splitting data into a training set and test set  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)  
X_train.head()  


# In[23]:


ins_fraud_num= X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',  
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',  
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',  
       'vehicle_claim']]  


# In[24]:


# Scaling the numeric values in the dataset  
  
scaler = StandardScaler()  
scaled_data = scaler.fit_transform(ins_fraud_num)  
scaled_ins_fraud_num = pd.DataFrame(data = scaled_data, columns = ins_fraud_num.columns, index = X_train.index)  
scaled_ins_fraud_num.head()  


# In[25]:


X_train.drop(columns = scaled_ins_fraud_num.columns, inplace = True)  
X_train = pd.concat([scaled_ins_fraud_num, X_train], axis = 1)  
X_train.head()  


# In[26]:


# Support Vector Classifier

svc = SVC()  
svc.fit(X_train, y_train)  
  
y_pred = svc.predict(X_test)  


# In[27]:


# accuracy_score, confusion_matrix and classification_report  

acc_svc_train = accuracy_score(y_train, svc.predict(X_train))  
acc_svc_test = accuracy_score(y_test, y_pred)  
  
print(f"Training accuracy of Support Vector Classifier is : {acc_svc_train}")  
print(f"Test accuracy of Support Vector Classifier is : {acc_svc_test}")  
  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[28]:


# KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 30)  
knn.fit(X_train, y_train)  
  
y_pred = knn.predict(X_test)  
# accuracy_score, confusion_matrix and classification_report  
  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
  
acc_knn_train = accuracy_score(y_train, knn.predict(X_train))  
acc_knn_test = accuracy_score(y_test, y_pred)  
  
print(f"Training accuracy of KNN is : {acc_knn_train}")  
print(f"Test accuracy of KNN is : {acc_knn_test}")  
  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[29]:


# Decision Tree Classifier

dt = DecisionTreeClassifier()  
dt.fit(X_train, y_train)  
  
y_pred = dt.predict(X_test)  
# accuracy_score, confusion_matrix and classification_report  
  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
  
acc_dt_train = accuracy_score(y_train, dt.predict(X_train))  
acc_dt_test = accuracy_score(y_test, y_pred)  
  
print(f"Training accuracy of Decision Tree is : {acc_dt_train}")  
print(f"Test accuracy of Decision Tree is : {acc_dt_test}")  
  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[30]:


# Random Forest Classifier

rfc = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)  
rfc.fit(X_train, y_train)  
  
y_pred = rfc.predict(X_test)  
# accuracy_score, confusion_matrix and classification_report  
  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
  
acc_rfc_train = accuracy_score(y_train, rfc.predict(X_train))  
acc_rfc_test = accuracy_score(y_test, y_pred)  
  
print(f"Training accuracy of Random Forest is : {acc_rfc_train}")  
print(f"Test accuracy of Random Forest is : {acc_rfc_test}")  
  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[31]:


# Gradient Boosting Classifier

gb = GradientBoostingClassifier()  
gb.fit(X_train, y_train)  
  
# accuracy score, confusion matrix, and classification report of gradient boosting classifier  
  
acc_gb = accuracy_score(y_test, gb.predict(X_test))  
  
print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")  
print(f"Test Accuracy of Gradient Boosting Classifier is {acc_gb} \n")  
  
print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")  
print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")


# In[ ]:




