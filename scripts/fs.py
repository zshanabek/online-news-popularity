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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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

scale_x.shape
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape
print(X_new)
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)

# rf = RandomForestClassifier(n_estimators=1100, criterion="entropy")
# clf_rf = rf.fit(X_train,y_train)
