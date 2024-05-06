# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:25:20 2023

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from numpy import unique
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
#Origin data
plt.figure(1)
data = pd.read_excel('data.xlsx')
#low+high
# X = data.values[1:,6:8]
# Y = data.values[1:,0]
# a = data.values[1:,5]

#low
X_low = data.values[1:,7]
X_high = data.values[1:,6]
Y = data.values[1:,5]
a = data.values[1:,0]

#low
X_high = X_high.reshape(-1,1)
X_low = X_low.reshape(-1,1)

# plt.scatter(X[:,0], X[:, 1],c=Y)
# plt.title("Original data",fontsize= 18)
# plt.xlabel("Low-sensitive",fontsize= 18)
# plt.ylabel("High-sensitive",fontsize= 18)
# plt.show()

#KMeans
plt.figure(1)
y_pred = KMeans(n_clusters=3).fit_predict(X_low)
result_kmeans = np.dstack((y_pred,Y,a))
# # high+
# y_pred = KMeans(n_clusters=3).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.title("K-Means",fontsize= 18)
# plt.ylabel("Low-sensitive",fontsize= 18)
# plt.xlabel("High-sensitive",fontsize= 18)
# plt.show()

# # #low
# plt.scatter(X[:, 0], Y, c=y_pred)
# plt.title("K-Means",fontsize= 18)
# plt.xlabel("Low-sensitive",fontsize= 18)
# plt.ylabel("concentration(mg/0.5ml)",fontsize= 18)
# plt.show()

plt.scatter(X_high,Y,label = 'High sensitive', marker='o')
plt.scatter(X_low,Y,label = 'Low sensitive',marker='^')

# # #Affinity Propagation Clustering
# # plt.figure(3)
# # y_pred = AffinityPropagation().fit_predict(X)
# # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# # plt.title("AffinityPropagation",fontsize= 18)
# # plt.xlabel("Low-sensitive",fontsize= 18)
# # plt.ylabel("High-sensitive",fontsize= 18)

# #mini-batch KMeans Clustering
# plt.figure(2)
# y_pred = MiniBatchKMeans(n_clusters=3,batch_size=14).fit_predict(X)
# result_mini = np.dstack((y_pred,Y,a))
# # #low+high
# # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# # plt.title("Mini-batch K-Means",fontsize= 18)
# # plt.xlabel("High-sensitive",fontsize= 18)
# # plt.ylabel("Low-sensitive",fontsize= 18)
# # plt.show()

# # #low
# plt.scatter(X[:, 0], Y, c=y_pred)
# plt.title("Mini-batch K-Means",fontsize= 18)
# plt.xlabel("Low-sensitive",fontsize= 18)
# plt.ylabel("concentration(mg/0.5ml)",fontsize= 18)
# plt.show()

# # #Aggregate clustering
# plt.figure(3)
# y_pred = AgglomerativeClustering(n_clusters=3,linkage='average').fit_predict(X)
# result_ag = np.dstack((y_pred,Y,a))
# # # low+high
# # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# # plt.title("AgglomerativeClustering",fontsize= 18)
# # plt.ylabel("Low-sensitive",fontsize= 18)
# # plt.xlabel("High-sensitive",fontsize= 18)

# # #low
# plt.scatter(X[:, 0], Y, c=y_pred)
# plt.title("AgglomerativeClustering",fontsize= 18)
# plt.xlabel("Low-sensitive",fontsize= 18)
# plt.ylabel("concentration(mg/0.5ml)",fontsize= 18)
# plt.show()


# # #Mean shift clustering
# # plt.figure(6)
# # y_pred = MeanShift().fit_predict(X)
# # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# # plt.title("MeanShift",fontsize= 18)
# # plt.xlabel("Low-sensitive",fontsize= 18)
# # plt.ylabel("High-sensitive",fontsize= 18)
# # plt.show()

# # #Gaussian Mixture Model
# # plt.figure(7)
# # y_pred = GaussianMixture(n_components=3).fit_predict(X)
# # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# # plt.title("GaussianMixture",fontsize= 18)
# # plt.xlabel("Low-sensitive",fontsize= 18)
# # plt.ylabel("High-sensitive",fontsize= 18)
# # plt.show()
