

import numpy as np
import pandas as pd

import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn

data  = vehicle = pd.read_csv('Dataset.csv')
data.shape

data.head()
data.tail()

data.describe()

vehicle_class = (vehicle['class'])
print('number of vehicle classes: ', np.unique(vehicle_class)) #takes out number of classes 
print('number of vehicle samples: ', len(vehicle_class))
print('number of vehicle samples in each class:', np.unique(vehicle_class, return_counts=True)) #its a balanced set 

from sklearn.model_selection import train_test_split
X = data.drop("class",axis=1)
y = data["class"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Decion Tree Accuracy:",accuracy_score(y_test,y_pred))



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))




#############

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))


from sklearn.svm import LinearSVC

svm_linear = LinearSVC(random_state=42, max_iter=5000)
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm_linear))





