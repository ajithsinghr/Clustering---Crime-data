# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:38:50 2023

@author: ramav
"""

# importing libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#==============================================================================

df = pd.read_csv("D:\\Assignments\\clustering\\crime_data.csv")
df.head()
df.shape
df.isnull().sum()

# renaming the column name for first column

df = df.rename(columns={df.columns[0]:"state"})
df.head()

#=============================================================================
# Exploratory Data Analysis(EDA)
# histogram
plt.hist(df["Murder"])
plt.hist(df["Assault"])
plt.hist(df["UrbanPop"])
plt.hist(df["Rape"])

# BOX PLOT
plt.boxplot(df['Murder'])
plt.boxplot(df['Assault'])
plt.boxplot(df['UrbanPop'])
plt.boxplot(df['Rape'])

# Scatter plot
plt.scatter(df['Murder'],df['Assault'])
plt.scatter(df['Assault'],df['UrbanPop'])
plt.scatter(df['UrbanPop'],df['Rape'])
plt.scatter(df['Murder'],df['Rape'])


# Normalization  

def norm(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


X = df.iloc[:,:]
# Normalized data frame (considering the numerical part of data)
ss_x = norm(df.iloc[:,1:])

#===============================================================================

# Hieraarchial or Aglomerative clustering
# Single linkage method

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,6))
plt.title("Crime rate dendogram")
dendo = shc.dendrogram(shc.linkage(ss_x,method='single'))

from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=4,linkage="single",affinity="euclidean")
y = agc.fit_predict(ss_x)
y = pd.DataFrame(y)
y.value_counts() # to know how many clusters are formed

clusterss = pd.DataFrame(y,columns=['clusters'])
ss_x["A_clustno"] = agc.labels_
ss_x["A_clustno"].value_counts()


#====================

# complete linkage method

plt.figure(figsize=(10,6))
plt.title("Crime rate dendogram")
dendo = shc.dendrogram(shc.linkage(ss_x,method='complete'))

from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=4,linkage="complete",affinity="euclidean")
y1 = agc.fit_predict(ss_x)
y1 = pd.DataFrame(y1)
y1.value_counts() # to know how many clusters are formed

clusterss = pd.DataFrame(y1,columns=['clusters'])
ss_x["A_clustno"] = agc.labels_
ss_x["A_clustno"].value_counts()

#==========================
# ward linkage method

plt.figure(figsize=(10,6))
plt.title("Crime rate dendogram")
dendo = shc.dendrogram(shc.linkage(ss_x,method='ward'))

from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")
y2 = agc.fit_predict(ss_x)
y2 = pd.DataFrame(y2)
y2.value_counts() # to know how many clusters are formed

clusterss = pd.DataFrame(y2,columns=['clusters'])
ss_x["A_clustno"] = agc.labels_
ss_x["A_clustno"].value_counts()

# Average 

plt.figure(figsize=(10,6))
plt.title("Crime rate dendogram")
dendo = shc.dendrogram(shc.linkage(ss_x,method='average'))

from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=4,linkage="average",affinity="euclidean")
y3 = agc.fit_predict(ss_x)
y3 = pd.DataFrame(y3)
y3.value_counts() # to know how many clusters are formed

clusterss = pd.DataFrame(y3,columns=['clusters'])
ss_x["A_clustno"] = agc.labels_
ss_x["A_clustno"].value_counts()



#===============================================================================
#===============================================================================

# K means clustering

from sklearn.cluster import KMeans
kmeans = KMeans()
kmeans.fit(ss_x)

# run in a loop to know the no of clusters needed

l1 = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(ss_x)
    l1.append(kmeans.inertia_)
    
print(l1)

pd.DataFrame(range(1,11))        
pd.DataFrame(l1)
    
pd.concat([pd.DataFrame(range(1,11)),pd.DataFrame(l1)], axis=1)

plt.scatter(range(1,11),l1)
plt.show()    

plt.plot(range(1,11),l1)
plt.xlabel("k value")
plt.ylabel("wcss value")
plt.show()

# elbow plot
#pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans,k=(1,11))
elbow.fit(ss_x)
elbow.poof()
plt.show()

# Updated model

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4,n_init=20)
kmeans.fit(ss_x)

kclus = kmeans.inertia_

df["cluster_name"]=kclus
df.head()

df.cluster_name.value_counts()

#=================================================================================
# DBSCAN

df.head()
x_ = df.iloc[:,1:5]

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(x_)


from sklearn.cluster import DBSCAN
dbs = DBSCAN(eps=0.75,min_samples=3)
dbs.fit(SS_X)

y = dbs.labels_
y = pd.DataFrame(y,columns=["Cluster"])
y["Cluster"].value_counts()

newdata = pd.concat([df,y],axis=1)

noisedata = newdata[newdata["Cluster"]==-1]
print(noisedata)
finaldata = newdata[newdata["Cluster"]==0] # excluding outliers or -1


noisedata.shape
finaldata.shape

df.shape





























