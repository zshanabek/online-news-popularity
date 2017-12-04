import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
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
X_train1, X_test, y_train1, y_test = model_selection.train_test_split(df[features], df['shares'], test_size=0.4, random_state=0)

scores_set = []
for smp in range(0,20):
    
    X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, random_state=smp, test_size=0.5)

    rf = RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion="entropy")

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    scores_set.append(accuracy_score(y_test,y_pred))

print scores_set

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion="entropy")

scores = []
scores = cross_val_score(rf, df[features], df['shares'], cv=5)

x =np.array(range(0,20))
z = np.array(list(scores_set))

plt.plot(x, z)
plt.xlabel('Value of random state')
plt.ylabel('Accuracy')
plt.show()
