#prepare a cluster of customers to predict the purchase power based on their income and spending score
#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans

#Loading the Dataset into DataFrame

df=pd.read_csv("Mall_Customers.csv")

#print(df.isna().sum(axis=0))# we found no missing values

X=df[["Annual Income (k$)","Spending Score (1-100)"]]

wcss_list=[]
for i in range(1,11):
    model =KMeans(n_clusters=i,init="k-means++",random_state=1)#we are finding elbow point 
    model.fit(X)
    wcss_list.append(model.inertia_)#will calculate wcss of each centroid then append it

#Visulaize the clusters
# plt.plot(range(1,11),wcss_list)
# plt.title("Elbow method graph")
# plt.xlabel("number of clusters")
# plt.ylabel("wcss")
# plt.show()
#we found elbow point at 2 so k=2

#training the model on our dataset
model=KMeans(n_clusters=5,init="k-means++",random_state=1)
y_predict=model.fit_predict(X)

print(y_predict)
#converting the Dataframe X into numpy array
X_array=X.values

#plotting to graph of clusters
plt.scatter(X_array[y_predict==0,0],X_array[y_predict==0,1],s=100,color="Green")
plt.scatter(X_array[y_predict==1,0],X_array[y_predict==1,1],s=100,color="Red")
plt.scatter(X_array[y_predict==2,0],X_array[y_predict==2,1],s=100,color="Yellow")
plt.scatter(X_array[y_predict==3,0],X_array[y_predict==3,1],s=100,color="Blue")
plt.scatter(X_array[y_predict==4,0],X_array[y_predict==4,1],s=100,color="Pink")

# plt.title("customer segmentation graph")
# plt.xlabel("Annual Income")
# plt.ylabel("spending score")
# plt.show()

joblib.dump(model,"Model.pkl")
print("model has been saved")