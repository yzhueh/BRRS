# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn import svm
from sklearn.metrics import f1_score,classification_report
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error,mean_squared_error
from sklearn import model_selection
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
n_classes = 5
classifier = svm.SVC(kernel='linear', C=1, gamma='auto')
data = pd.read_excel('data.xlsx')
X = data.values[1:,5:7]
Y = data.values[1:,0]
Con = data.values[1:,0]

skf=StratifiedKFold(n_splits=4)
skf.get_n_splits(X,Y)

X_train, X_test, Con_train, Con_test = model_selection.train_test_split(X, Con,test_size= 0.3)



    
# Regression
model1 = Ridge(alpha=0.5)
model2 = SVR()
model3 = LassoLarsCV()
model4 = LassoCV()
model5 = RandomForestRegressor()
model6 = DecisionTreeRegressor()
model7 = LinearRegression()
model8 = LogisticRegression()

# model1.fit(X_train, Con_train)
# Con_pred = model1.predict(X_test)

# #Plot
# plt.figure(1)
# plt.scatter(Con_test, Con_pred)
# plt.xlabel('label',fontsize= 18)
# plt.ylabel('predict',fontsize= 18)
# plt.title("Ridge",fontsize= 18)
# x1, y1 = 0, 0
# x2, y2 = 8, 8
# plt.plot([x1, x2], [y1, y2])
 
###############################SVR
model2.fit(X_train, Con_train)
Con_pred = model2.predict(X_test)
result_svr = np.dstack((Con_pred,Con_test))

# #Plot
# plt.figure(2)
# plt.scatter(Con_test, Con_pred)
# plt.xlabel('True value',fontsize= 18)
# plt.ylabel('Predict',fontsize= 18)
# plt.title("SVR",fontsize= 18)
# x1, y1 = 0, 0
# x2, y2 = 8, 8
# plt.plot([x1, x2], [y1, y2])
# EVS_score = explained_variance_score(Con_test, Con_pred)
# MAE_score = mean_absolute_error(Con_test, Con_pred)
# MSE_score = mean_squared_error(Con_test, Con_pred)
# print("SVR")
# print("EVS =",EVS_score)
# print("MAE =",MAE_score)
# print("MSE =",MSE_score)

# model3.fit(X_train, Con_train)
# Con_pred = model3.predict(X_test)

# #Plot
# plt.figure(3)
# plt.scatter(Con_test, Con_pred)
# plt.xlabel('label',fontsize= 18)
# plt.ylabel('predict',fontsize= 18)
# plt.title("LassoLarsCV",fontsize= 18)
# x1, y1 = 0, 0
# x2, y2 = 8, 8
# plt.plot([x1, x2], [y1, y2])


# model4.fit(X_train, Con_train)
# Con_pred = model4.predict(X_test)

# #Plot
# plt.figure(4)
# plt.scatter(Con_test, Con_pred)
# plt.xlabel('label',fontsize= 18)
# plt.ylabel('predict',fontsize= 18)
# plt.title("LassoCV",fontsize= 18)
# x1, y1 = 0, 0
# x2, y2 = 8, 8
# plt.plot([x1, x2], [y1, y2])



# # ##############RandomForestRegressor
model5.fit(X_train, Con_train)
Con_pred = model5.predict(X_test)
result_rf = np.dstack((Con_pred,Con_test))

# #Plot
# # plt.figure(5)
# plt.scatter(Con_test, Con_pred)
# plt.xlabel('True value',fontsize= 18)
# plt.ylabel('Predict',fontsize= 18)
# plt.title("RandomForestRegressor",fontsize= 18)
# x1, y1 = 0, 0
# x2, y2 = 8, 8
# plt.plot([x1, x2], [y1, y2])
EVS_score = explained_variance_score(Con_test, Con_pred)
MAE_score = mean_absolute_error(Con_test, Con_pred)
MSE_score = mean_squared_error(Con_test, Con_pred)
print("RandomForest")
print("EVS =",EVS_score)
print("MAE =",MAE_score)
print("MSE =",MSE_score)

# #############DecisionTreeRegressor
model6.fit(X_train, Con_train)
Con_pred = model6.predict(X_test) 
result_dt = np.dstack((Con_pred,Con_test))

# #Plot
# plt.figure(6)
# plt.scatter(Con_test, Con_pred)
# plt.xlabel('True value',fontsize= 18)
# plt.ylabel('Predict',fontsize= 18)
# plt.title("DecisionTreeRegressor",fontsize= 18)
# x1, y1 = 0, 0
# x2, y2 = 8, 8
# plt.plot([x1, x2], [y1, y2]) 
EVS_score = explained_variance_score(Con_test, Con_pred)
MAE_score = mean_absolute_error(Con_test, Con_pred)
MSE_score = mean_squared_error(Con_test, Con_pred)
print("DecisionTree")
print("EVS =",EVS_score)
print("MAE =",MAE_score)
print("MSE =",MSE_score)


# ##############LinearRegression()
model7.fit(X_train, Con_train)
Con_pred = model7.predict(X_test) 
result_lr = np.dstack((Con_pred,Con_test))

# #Plot
# plt.figure(7)
# plt.scatter(Con_test, Con_pred)
# plt.xlabel('True value',fontsize= 18)
# plt.ylabel('Predict',fontsize= 18)
# plt.title("LinearRegression",fontsize= 18)
# x1, y1 = 0, 0
# x2, y2 = 8, 8
# plt.plot([x1, x2], [y1, y2]) 
# EVS_score = explained_variance_score(Con_test, Con_pred)
# MAE_score = mean_absolute_error(Con_test, Con_pred)
# MSE_score = mean_squared_error(Con_test, Con_pred)
# print("LinearRegression")
# print("EVS =",EVS_score)
# print("MAE =",MAE_score)
# print("MSE =",MSE_score)




# model8.fit(X_train, Con_train)
# Con_pred = model8.predict(X_test)

# #Plot
# plt.figure(8)
# plt.scatter(Con_test, Con_pred)
# plt.xlabel('label',fontsize= 18)
# plt.ylabel('predict',fontsize= 18)
# plt.title("LogisticRegression",fontsize= 18)
# x1, y1 = 0, 0
# x2, y2 = 8, 8
# plt.plot([x1, x2], [y1, y2]) 

# print(skf)

# for train_index,test_index in skf.split(X,Y):

#     # print("Train Index:",train_index,",Test Index:",test_index)

#     X_train,X_test=X[train_index],X[test_index]
#     Y_train,Y_test=Y[train_index],Y[test_index]
#     Con_train,Con_test=Con[train_index],Con[test_index]
    
#     # classifier.fit(X_train, Y_train)
#     # Y_pred = classifier.predict(X_test)
    
#     # # f1 = f1_score(Y_test, Y_pred,average='macro')
#     # # print('F1-score:', f1)
    
#     # target_names = ['class 0', 'class 1', 'class 2', 'class 3']
#     # print(classification_report(Y_test, Y_pred, target_names=target_names))
    
    
# #Regression
#     # model = Ridge(alpha=0.5)
#     model = SVR()
#     model.fit(X_train, Con_train)
#     Con_pred = model.predict(X_test)
    
#     EVS_score = explained_variance_score(Con_test, Con_pred)
#     MAE_score = mean_absolute_error(Con_test, Con_pred)
#     MSE_score = mean_squared_error(Con_test, Con_pred)
    
#     print("EVS =",EVS_score)
#     print("MAE =",MAE_score)
#     print("MSE =",MSE_score)
