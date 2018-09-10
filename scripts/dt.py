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
import xlsxwriter
import matplotlib.pyplot as plt
from sklearn import tree

csv_filename="OnlineNewsPopularity.csv"

df=pd.read_csv(csv_filename)

popular = df.shares >= 1400	
unpopular = df.shares < 1400

df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0

features=list(df.columns[2:60])

X = df[features]
y = df['shares']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)

rf = tree.DecisionTreeClassifier(criterion="entropy")
clf = rf.fit(X_train,y_train)

tree.export_graphviz(clf, out_file='tree.dot')     