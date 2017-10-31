import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from time import time
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score
import matplotlib.pyplot as plt

csv_filename="OnlineNewsPopularity.csv"

df=pd.read_csv(csv_filename)

popular = df.shares >= 1400	
unpopular = df.shares < 1400

df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0


features=list(df.columns[2:60])
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], df['shares'], test_size=0.15, random_state=0)

k_range = range(1,1300,100)
k_scores = []

iteration = 1
cls = []

for k in k_range:
    rf = RandomForestClassifier(n_estimators=k)
    scores = cross_val_score(rf, X_train, y_train)
    k_scores.append(scores.mean())
    print k
    print scores.mean()

print(k_scores)




# plt.plot(x, z)
# plt.xlabel('Value of K for Random Forest')
# plt.ylabel('Cross-validated accuracy')
# plt.show()
# y_pred = clfs[40].predict(X_test)
# print(accuracy_score(y_test, y_pred))