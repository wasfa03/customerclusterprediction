import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
import streamlit as st

kmeans=joblib.load("Model.pkl")
df=pd.read_csv("Mall_Customers.csv")
X=df[["Annual Income (k$)","Spending Score (1-100)"]]
X_array=X.values

st.set_page_config(page_title="Customer Cluster Prediction", layout="centered")
st.title("Customer Cluster Prediction")
st.write("Enter the Customer Annual Income and Spending score to predict the cluster:")

#inputs
annual_income=st.number_input("Annual Income of a Customer",min_value=0,max_value=400,value=50)
spending_score=st.slider("Spending Score between 1-100",1,100,20)

if st.button("Predict Cluster"):
    input_data=np.array([[annual_income,spending_score]])
    cluster=kmeans.predict(input_data)[0]
    st.success(f"Predicted Cluster is: {cluster}")