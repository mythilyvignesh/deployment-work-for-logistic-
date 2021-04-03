#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
st.title('model deployement:logistic regression')
st.sidebar.header('User input parameter')
def user_input_features():
    Daily_Time_Spent_on_Site = st.sidebar.number_input(" Daily Time Spent on Site")
    Age = st.sidebar.number_input("insert age")
    Area_Income = st.sidebar.number_input("Area Income")
    Daily_internet_usage = st.sidebar.number_input("Daily internet usage")
    Male = st.sidebar.selectbox('male',('1','0'))    
    data = {'Daily_Time_Spent_on_Site':Daily_Time_Spent_on_Site,
           'Age':Age,
           'Area_Income':Area_Income,
           'Daily_internet_usage':Daily_internet_usage,
           'Male':Male}
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_features()
st.subheader('User input parameters')
st.write(df)
A=pd.read_csv('D:\\jup notebook\\ml\\logistic\\advertising.csv')
A.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1,inplace=True)
A=A.dropna()
x = A[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage','Male']]
y = A[['Clicked on Ad']]
logmodel = LogisticRegression()
logmodel.fit(x,y)
predictions = logmodel.predict(df)
prediction_proba = logmodel.predict_proba(df)
st.subheader('Predicted result')
st.write('Yes' if predictions == 0 else 'No')
st.subheader('Prediction probability')
st.write(prediction_proba)


# In[ ]:





# In[ ]:




