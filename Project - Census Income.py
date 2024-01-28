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


Census_Income = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/census_income.csv')
Census_Income.head()


# In[3]:


Census_Income.shape


# In[4]:


Census_Income.info()


# In[5]:


Census_Income.dtypes.value_counts()


# In[6]:


Census_Income.isna().sum()


# In[7]:


Census_Income.columns


# In[8]:


Census_Income.describe()


# In[9]:


Census_Income.dropna(inplace = True)
Census_Income.drop_duplicates(inplace=True)


# In[10]:


print(Census_Income['Income'].value_counts())
Census_Income['Income'].value_counts().plot.pie(autopct= '%1.1f%%')


# Adults with income less than 50000 are almost 3 times greater than adults with income greater than 50000.

# In[11]:


Census_Income['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Count')


# Age attribute is right-skewed and not symetric. min and max age in btw 17 to 90.

# In[12]:


Census_Income['Fnlwgt'].hist(figsize=(6,4))
plt.xlabel('Fnlwgt')
plt.ylabel('Count')


# Distribution is Rightly skewed

# In[13]:


Census_Income['Capital_gain'].hist(figsize=(6,4))
plt.xlabel('Capital_gain')
plt.ylabel('Count')


# Capital-gain shows that either a person has no gain or has gain of very large amount(10k or 99k).

# In[14]:


Census_Income['Capital_loss'].hist(figsize=(6,4))
plt.xlabel('Capital_loss')
plt.ylabel('Count')


# It is also similar to captain-gain we can also remove this feature too. as most of the rows have value zero.

# In[15]:


Census_Income['Hours_per_week'].hist(figsize=(6,4))
plt.xlabel('Hours_per_week')
plt.ylabel('Count')


# In[16]:


plt.figure(figsize=(12,4))
total = float(len(Census_Income['Income']))
a = sns.countplot(x='Workclass',data=Census_Income)

for f in a.patches:
    height = f.get_height()
    a.text(f.get_x() + f.get_width()/2., height+3, '{:1.2f}'.format((height/total)*100),ha="center")


# Most of them are form private workclass with around 74% and rest all have less contribution compare to private with without-pay as least count of around 0.05% of total count.

# In[17]:


plt.figure(figsize=(18,4))

a= float(len(['Income']))

a= sns.countplot(x='Education',data=Census_Income)
for s in a.patches:
    height = s.get_height()
    a.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')


# HS-grad has highest no of adults and preschool have lowest.

# In[18]:


plt.figure(figsize=(15,4))
total = float(len(Census_Income) )
a = sns.countplot(x="Marital_status", data=Census_Income)
for p in a.patches:
    height = p.get_height()
    a.text(p.get_x()+p.get_width()/2.,
            height + 3,'{:1.2f}'.format((height/total)*100),ha="center")


# Married-civ-spouse has maximum number of samples. Married-AF-spouse has minimum number of obs.

# In[19]:


plt.figure(figsize=(15,4))
total = float(len(Census_Income) )
a = sns.countplot(x="Occupation", data=Census_Income)
for p in a.patches:
    height = p.get_height()
    a.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format((height/total)*100),ha="center")
a.set_xticklabels(a.get_xticklabels(), rotation=60)
plt.show()


# Craft-repair has the maximum count. Armed-Forces has minimum samples in the occupation attribute.

# In[20]:


plt.figure(figsize=(12,4))
total = float(len(Census_Income) )
a = sns.countplot(x="Relationship", data=Census_Income)
for p in a.patches:
    height = p.get_height()
    a.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format((height/total)*100),ha="center")


# In[21]:


plt.figure(figsize=(5,4))
total = float(len(Census_Income) )
a = sns.countplot(x="Income", data=Census_Income)
for p in a.patches:
    height = p.get_height()
    a.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format((height/total)*100),ha="center")


# In income there is 2 group,group1(who earns more than 50k) 24.90% belong to income and group2(who earns less than 50k) 75.10% belong to income.

# In[22]:


plt.figure(figsize=(5,4))
sns.boxplot(x='Income',y='Age',data=Census_Income).set_title('Box plot of INCOME and AGE')


# We can observe that the median age for people earning more than 50k is significantly greater than the median of people earning less than 50k. So, older people are more likely to earn more than $50k a year as compared to their younger counterparts.

# In[23]:


plt.figure(figsize=(8,4))
a=sns.countplot(x='Workclass',hue ='Income',data=Census_Income).set_title("workclass vs count")


# In All the workclasses number of people earning less then 50k are more then those earning 50k.

# In[24]:


plt.figure(figsize=(10,4))
sns.countplot(x="Relationship", hue="Income",data=Census_Income);


# Mostly a person with relation as husband in a family has most count of people with more then 50k income.

# In[25]:


plt.figure(figsize=(10,4))
Census_Income['Sex'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#33cccc','#ded033'])


# In[26]:


plt.figure(figsize=(5,4))
sns.countplot(x="Sex", hue="Income",data=Census_Income);


# In[27]:


Census_Income.drop(['Education','Capital_gain','Capital_loss'], axis = 1, inplace = True)
Census_Income.shape


# In[28]:


le = LabelEncoder()
for column in Census_Income.columns:
    if Census_Income[column].dtype == 'object':
        Census_Income[column] = le.fit_transform(Census_Income[column])

Census_Income.head()


# In[29]:


X = Census_Income.drop('Income', axis = 1)
y = Census_Income['Income']
X = StandardScaler().fit_transform(X)
X.shape


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=136)


# In[31]:


modelLG = LogisticRegression()
modelLG.fit(x_train, y_train)
y_predict = modelLG.predict(x_test)
print('Accuracy=',accuracy_score(y_test, y_predict))


# In[32]:


print(classification_report(y_test,y_predict))


# In[33]:


cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[34]:


accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1_score = f1_score(y_test, y_predict)


# In[35]:


# Create a bar plot to visualize the metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1_score]

plt.figure(figsize=(10, 6))
sns.barplot(x=values, y=metrics, palette='viridis')
plt.title('Model Evaluation Metrics')
plt.xlabel('Value')
plt.xlim(0, 1)  # Adjust the x-axis limits if needed
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Annotate the values on the bars
for i, v in enumerate(values):
    plt.text(v + 0.01, i, f'{v:.2f}', color='black', va='center')

plt.show()


# In[ ]:




