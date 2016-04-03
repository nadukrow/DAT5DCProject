# -*- coding: utf-8 -*-
"""

@author: nadukrow
"""

import pandas as pd

train = pd.read_csv('training_data.csv')
train.shape #1000 rows, 6 columns
train.columnns #Prints Patient ID, Resp, PR Seq, RT Seq, Viral load count, CD4+ count

train.Resp.value_counts() #To count the number of positive vs negative prognosis after treatment. 794 neg 206 pos.

train[train.Resp==0].describe() #Compare the VL and CD4 count of all respondents with 0
train[train.Resp==1].describe() #Compare the VL and CD4 count of all respondents with 1

train[['Resp', 'VL-t0', 'CD4-t0']][:] #This is to look the the indicators with as well as the associated response.
indicators_col = train[['Resp', 'VL-t0', 'CD4-t0']][:] #This is to solely focus our attention to the indicators and their associated reponse.

indicators_col[train.Resp==0].mean() #We can see the avg VL and CD4 count for respondents with 0.
indicators_col[train.Resp==1].mean() #See the same with respondents who had a positive prognosis.

neg = indicators_col[train.Resp==0]
pos = indicators_col[train.Resp==1]

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics

train = pd.read_csv('training_data.csv')

#Logistic Regression
feature_cols = ['VL-t0']
X = train[feature_cols]
y = train.Resp

#Train test split method using accuracy for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

#Evaluation
print metrics.accuracy_score(y_test, y_pred)
print metrics.confusion_matrix(y_test, y_pred)
print metrics.roc_auc_score(y_test, y_prob)       
print metrics.log_loss(y_test, y_prob) 

#Cross Validation method using AUC for scoring
logreg = LogisticRegression()
score = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

#Logistic Regression
feature_cols = ['CD4-t0']
X = train[feature_cols]
y = train.Resp

#Train test split method using accuracy for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

#Evaluation
print metrics.accuracy_score(y_test, y_pred)
print metrics.confusion_matrix(y_test, y_pred)
print metrics.roc_auc_score(y_test, y_prob)       
print metrics.log_loss(y_test, y_prob) 

#Cross Validation method using AUC for comparison
logreg = LogisticRegression()
score = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

#Logistic Regression
feature_cols = ['VL-t0', 'CD4-t0']
X = train[feature_cols]
y = train.Resp

#Train test split method using accuracy for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

#Evaluation
print metrics.accuracy_score(y_test, y_pred)
print metrics.confusion_matrix(y_test, y_pred)
print metrics.roc_auc_score(y_test, y_prob)       
print metrics.log_loss(y_test, y_prob) 

#Cross Validation method using AUC for scoring
logreg = LogisticRegression()
score = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

#Apply prediction on the test data
test = pd.read_csv('test_data.csv')
train = pd.read_csv('training_data.csv')

#Viral Load feature
feature_cols = ['VL-t0']
X = train[feature_cols]
y = train.Resp

logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X_test)
probs = logreg.predict_proba(X_test)[:, 1]

print metrics.accuracy_score(y_test, y_pred)
print metrics.roc_auc_score(y_test, y_prob)  

feature_cols = ['CD4-t0']
X = train[feature_cols]
y = train.Resp

logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X_test)

print metrics.accuracy_score(y_test, y_pred)

feature_cols = ['VL-t0', 'CD4-t0']
X = train[feature_cols]
y = train.Resp

logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X_test)

print metrics.accuracy_score(y_test, y_pred)
