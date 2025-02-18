#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from  sklearn.ensemble import RandomForestClassifier


# In[3]:


df_train = pd.read_csv(r"C:\Users\KHUSHI\Downloads\archive (4)\fraudTrain.csv", header = 0)
df_test = pd.read_csv(r"C:\Users\KHUSHI\Downloads\archive (4)\fraudTest.csv", header = 0)


# In[4]:


df_train.head()


# In[5]:


df_test.head()


# In[6]:


df_train.tail()


# In[7]:


df_train.shape


# In[8]:


df_test.shape


# In[9]:


df_train.size


# In[10]:


df_test.size


# In[11]:


df_train.info()


# In[12]:


df_test.info()


# In[13]:


df_train.describe()


# In[14]:


df_test.describe()


# In[15]:


df_train.isnull().values.any()


# In[16]:


df_test.isnull().values.any()


# In[17]:


df_train.count()


# In[18]:


df_test.count()


# In[19]:


df_combined = pd.concat([df_train, df_test], axis = 0)


# In[20]:


df_combined.head()


# In[21]:


df_combined.shape


# In[22]:


df_combined.size


# In[23]:


df_combined.info()


# In[24]:


df_combined.drop(labels = ["first", "last", "job", "dob", "trans_num", "street", "trans_date_trans_time","city","state"], axis = 1, inplace = True)


# In[25]:


df_combined.head()


# In[26]:


sns.countplot(x='gender', data=df_combined)
plt.title("Gender Distribution")
plt.show()


# In[48]:


correlation_matrix = df_combined.corr()
plt.figure(figsize = (12,8))
sns.heatmap(correlation_matrix, cmap = 'coolwarm', annot = False,  fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[29]:


encoder = LabelEncoder()
new_col = encoder.fit_transform(df_combined["merchant"].values)
df_combined["merchant_new"] = new_col
df_combined.drop(labels = ["merchant"], axis = 1, inplace = True)


# In[30]:


ncoder = LabelEncoder()
new_col1 = encoder.fit_transform(df_combined["category"].values)
df_combined["category_new"] = new_col1
df_combined.drop(labels = ["category"], axis = 1, inplace = True)


# In[31]:


df_combined = pd.get_dummies(df_combined)
df_combined.drop(labels=['gender_F'], axis = 1, inplace = True)


# In[32]:


df_combined.head()


# In[33]:


X = df_combined.drop("is_fraud", axis = 1)
y = df_combined["is_fraud"]


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)


# In[36]:


lr_model = LogisticRegression()


# In[37]:


lr_model.fit(X_train, y_train)


# In[38]:


lr_predictions = lr_model.predict(X_test)


# In[39]:


print("Logistic Regression Model: ")
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))
print("Accuracy: ", accuracy_score(y_test, lr_predictions))


# In[40]:


dt_model = DecisionTreeClassifier()


# In[41]:


dt_model.fit(X_train, y_train)


# In[42]:


dt_predictions = dt_model.predict(X_test)


# In[43]:


print("Decision Tree Model: ")
print(confusion_matrix(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions))
print("Accuracy: ", accuracy_score(y_test, dt_predictions))


# In[44]:


rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)


# In[45]:


rf_model.fit(X_train, y_train)


# In[46]:


rf_predictions = rf_model.predict(X_test)


# In[47]:


print("Random Forest Model: ")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))
print("Accuracy: ", accuracy_score(y_test, rf_predictions))


# In[ ]:




