#!/usr/bin/env python
# coding: utf-8

# # Importing Datasets

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Data Collections & Data Processing

# In[4]:


# Loading the data set to p pandas DataFrame

sonar_data = pd.read_csv("sonar data.csv",header=None)
sonar_data


# In[6]:


# Number of rows & columns

sonar_data.shape


# In[9]:


# Describe give statistical data

sonar_data.describe()


# In[11]:


sonar_data[60].value_counts()


# ### R -> Rock
# ### M -> Mine

# In[12]:


sonar_data.groupby(60).mean() 


# In[13]:


# Separating data and labels

X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60]


# In[15]:


print(X)
print(Y)


# # Training & Test Data

# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)


# In[17]:


print(X.shape,X_train.shape,X_test.shape)


# In[20]:


print(X_train)
print(Y_train)


# # Model Training -> Logistic Regression

# In[18]:


model = LogisticRegression()


# In[21]:


# Training the logistic regression model with training data

model.fit(X_train, Y_train)


# # Model Evaluations

# In[22]:


# Accuracy on the training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[23]:


print('Accuracy on trainingdata :',training_data_accuracy)


# In[24]:


# Accuracy on the test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[25]:


print('Accuracy on Test data :',test_data_accuracy)


# # Making Predictive System

# In[32]:


input_data = (0.0516,0.0944,0.0622,0.0415,0.0995,0.2431,0.1777,0.2018,0.2611,0.1294,0.2646,0.2778,0.4432,0.3672,0.2035,0.2764,0.3252,0.1536,0.2784,0.3508,0.5187,0.7052,0.7143,0.6814,0.5100,0.5308,0.6131,0.8388,0.9031,0.8607,0.9656,0.9168,0.7132,0.6898,0.7310,0.4134,0.1580,0.1819,0.1381,0.2960,0.6935,0.8246,0.5351,0.4403,0.6448,0.6214,0.3016,0.1379,0.0364,0.0355,0.0456,0.0432,0.0274,0.0152,0.0120,0.0129,0.0020,0.0109,0.0074,0.0078)
# Changing the input_data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the numpy as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if(prediction[0]=='R'):
    print('Object is a Rock')
else:
    print('Object is a Mine')


# In[ ]:




