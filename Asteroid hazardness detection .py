#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import seaborn as sns


# In[2]:


df = pd.read_csv("G:/DATA SETT/nasa.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df = df.drop(['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date',
                        'Epoch Date Close Approach', 'Orbit Determination Date'] , axis = 1)
df.head()


# In[6]:


harzads_label=pd.get_dummies(df['Hazardous'])
harzads_label


# In[7]:


df = pd.concat([df,harzads_label], axis = 1)
df.head()


# In[8]:


df.info()


# In[9]:


df['Equinox'].value_counts


# In[10]:


df['Orbiting Body'].value_counts()


# In[11]:


df=df.drop(['Equinox','Orbiting Body'], axis=1)


# In[12]:


mean_value = df['Absolute Magnitude'].mean()


# In[13]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[14]:


df = df.drop(['Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)'
             ,'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 
             'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)', 
             'Miss Dist.(kilometers)', 'Miss Dist.(miles)'], axis = 1)
df.head()


# In[15]:


plt.figure(figsize = (20,20))
sns.heatmap(df.corr(),annot = True)


# In[16]:


sns.clustermap(df.corr(), annot=True)
plt.show()


# In[ ]:


df.drop([False], axis = 1, inplace = True)


# In[19]:


df.head()


# In[20]:


# Model Buildind
x = df.drop([True], axis = 1)
y = df[True].astype(int)


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0 , test_size = 0.3)


# In[29]:


get_ipython().system('pip install xgboost')


# In[30]:


from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance

xbg_model = XGBClassifier()
xbg_model.fit(x_train, y_train)
plot_importance(xbg_model)
pyplot.show()


# In[32]:


get_ipython().system('pip install scikit-learn')


# In[36]:


from sklearn.metrics import accuracy_score
predictions = xbg_model.predict(x_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')


# In[ ]:




