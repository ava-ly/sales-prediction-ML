#!/usr/bin/env python
# coding: utf-8

# # Future Sales Prediction with Machine Learning

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


data = pd.read_csv("../data/advertising.csv")
print(data.head())


# In[3]:


print(data.isnull().sum())


# In[5]:


import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()


# In[ ]:


figure = px.scatter(data_frame = data, x="Sales",
                    y="Newspaper", size="Newspaper", trendline="ols")
figure.show()


# In[ ]:


figure = px.scatter(data_frame = data, x="Sales",
                    y="Radio", size="Radio", trendline="ols")
figure.show()


# In[8]:


correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))


# ## Future Sales Prediction Model

# In[10]:


x = np.array(data.drop(["Sales"], axis=1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[11]:


model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[12]:


#features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))


# In[ ]:




