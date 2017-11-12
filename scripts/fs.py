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
from sklearn.svm import SVC
from sklearn import preprocessing
from time import time
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score

csv_filename="OnlineNewsPopularity.csv"

df=pd.read_csv(csv_filename)

popular = df.shares >= 1400	
unpopular = df.shares < 1400

df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0

features=list(df.columns[2:60])

X = df[features]
norm_x = preprocessing.normalize(X)
scale_x = preprocessing.scale(norm_x)
y = df['shares']

X_train, X_test, y_train, y_test = model_selection.train_test_split(scale_x, y, test_size=0.4)

print "RandomForest scaled"
rf = RandomForestClassifier(n_estimators=1100, criterion="entropy")
clf_rf = rf.fit(X_train,y_train)
# score_rf=clf_rf.score(X_test,y_test)
# print "Acurracy: ", score_rf

model = SelectFromModel(rf, prefit=True)
X_new = model.transform(X_train)             

print(X_new)
print('Shape')
print(X_new.shape)
