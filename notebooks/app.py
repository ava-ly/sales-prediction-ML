import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Title of the app
st.title("Advertising Data Analysis & Sales Prediction")

# Load Data
st.header("Dataset Overview")
data = pd.read_csv("../data/advertising.csv")
st.write(data.head())

# Check for missing values
st.subheader("Missing Values")
st.write(data.isnull().sum())

# Visualizations
st.header("Data Visualization")

# TV vs Sales
st.subheader("TV Advertising vs Sales")
fig_tv = px.scatter(data_frame=data, x="Sales", y="TV", size="TV", trendline="ols", title="TV vs Sales")
st.plotly_chart(fig_tv)

# Newspaper vs Sales
st.subheader("Newspaper Advertising vs Sales")
fig_newspaper = px.scatter(data_frame=data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols", title="Newspaper vs Sales")
st.plotly_chart(fig_newspaper)

# Radio vs Sales
st.subheader("Radio Advertising vs Sales")
fig_radio = px.scatter(data_frame=data, x="Sales", y="Radio", size="Radio", trendline="ols", title="Radio vs Sales")
st.plotly_chart(fig_radio)

# Correlation
st.header("Correlation with Sales")
correlation = data.corr()
st.write(correlation["Sales"].sort_values(ascending=False))

# Build Prediction Model
st.header("Sales Prediction Model")

# Splitting data
x = np.array(data.drop(["Sales"], axis=1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(xtrain, ytrain)
accuracy = model.score(xtest, ytest)
st.subheader(f"Model Accuracy: {accuracy:.2f}")

# User Inputs for Prediction
st.subheader("Predict Future Sales")
tv_input = st.number_input("TV Advertising Budget", min_value=0.0, value=230.1)
radio_input = st.number_input("Radio Advertising Budget", min_value=0.0, value=37.8)
newspaper_input = st.number_input("Newspaper Advertising Budget", min_value=0.0, value=69.2)

# Make Prediction
if st.button("Predict Sales"):
    features = np.array([[tv_input, radio_input, newspaper_input]])
    prediction = model.predict(features)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")
