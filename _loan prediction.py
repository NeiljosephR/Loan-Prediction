#!/usr/bin/env python
# coding: utf-8

# # IMPORT THE REQUIRED LIBRARIES

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # READ THE DATASET

# In[52]:


train=pd.read_csv("train.csv")
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})
train.isnull().sum()


# # TRAINING THE DATA

# In[53]:


Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv('test.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()


# In[54]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data.Dependents.dtypes


# In[7]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[8]:


data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


# In[9]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[10]:


data.Married=data.Married.map({'Yes':1,'No':0})


# In[11]:


data.Married.value_counts()


# In[12]:


data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})


# In[13]:


data.Dependents.value_counts()


# In[14]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[15]:


data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})


# In[16]:


data.Education.value_counts()


# In[17]:


data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})


# In[18]:


data.Self_Employed.value_counts()


# In[19]:


data.Property_Area.value_counts()


# In[20]:


data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})


# In[21]:


data.Property_Area.value_counts()


# In[22]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[23]:


data.head()


# In[24]:


data.Credit_History.size


# In[25]:


data.Credit_History.fillna(np.random.randint(0,2),inplace=True)

data.isnull().sum()


# In[26]:


data.Married.fillna(np.random.randint(0,2),inplace=True)

data.isnull().sum()


# In[27]:


data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)


# In[28]:


data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)

data.isnull().sum()


# In[29]:


data.Gender.value_counts()


# In[30]:


from random import randint 
data.Gender.fillna(np.random.randint(0,2),inplace=True)

data.Gender.value_counts()


# In[31]:


data.Dependents.fillna(data.Dependents.median(),inplace=True)

data.isnull().sum()


# In[32]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[33]:


data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)

data.isnull().sum()


# In[34]:


data.head()


# In[35]:


data.drop('Loan_ID',inplace=True,axis=1)

data.isnull().sum()


# In[36]:


train_X=data.iloc[:614,]
train_y=Loan_status
X_test=data.iloc[614:,]
seed=7


# # TRAINING AND TESTNG

# In[37]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=seed)


# # IMPORTING LIBRARIES FROM SKLEARN

# In[38]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[39]:


models=[]
models.append(("logreg",LogisticRegression()))
models.append(("tree",DecisionTreeClassifier()))
models.append(("lda",LinearDiscriminantAnalysis()))
models.append(("svc",SVC()))
models.append(("knn",KNeighborsClassifier()))
models.append(("nb",GaussianNB()))


# In[40]:


seed=7
scoring='accuracy'


# In[41]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
names=[]


# In[42]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=LogisticRegression()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# # SVC PREDICT

# In[43]:


df_output=pd.DataFrame()

outp=svc.predict(X_test).astype(int)
outp


# In[44]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[45]:


df_output.head()


# In[46]:


df=df_output[['Loan_ID','Loan_Status']].to_csv('output.csv',index=False)
pd.read_csv('output.csv')


# # PREDICTING STATUS

# In[47]:


print(pred[5])


# In[48]:


print(pred[9])


# In[ ]:




