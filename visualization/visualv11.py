# -*- coding: utf-8 -*-
"""

@author: nadukrow
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

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

pos.plot(kind='scatter', x='VL-t0', y='CD4-t0', alpha=0.3)
plt.show()

neg.plot(kind='scatter', x='VL-t0', y='CD4-t0', alpha=0.3)
plt.show()

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

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

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

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

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

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')