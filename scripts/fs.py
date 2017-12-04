from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from time import time
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

csv_filename="OnlineNewsPopularity.csv"

df=pd.read_csv(csv_filename)

popular = df.shares >= 1400	
unpopular = df.shares < 1400

df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0

features=list(df.columns[2:60])

X = df[features]

y = df['shares']


clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)          

feature_names = X.columns.values
mask = model.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
print(new_features)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, df['shares'], test_size=0.4, random_state=0)


rf = RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion="gini")

scores = []
scores = cross_val_score(rf, df[features], df['shares'], cv=5, n_jobs=-1)


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(scores)