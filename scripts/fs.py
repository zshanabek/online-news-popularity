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
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
csv_filename="OnlineNewsPopularity.csv"

df=pd.read_csv(csv_filename)
plt.matshow(df.corr())
popular = df.shares >= 1400	
unpopular = df.shares < 1400

df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0

features=list(df.columns[2:60])

X = df[features]

y = df['shares']

# a  = list(df[features].columns.values)
# for i in range(0, len(a)):
#     a[i] = str(i)+" " +a[i]

# print(a) 
#    
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion="gini")

rf = rf.fit(X, y)
importances = rf.feature_importances_
print(rf.feature_importances_)  

std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), df[features].columns.values[indices],rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

model = SelectFromModel(rf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
feature_names = X.columns.values
mask = model.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, df['shares'], test_size=0.4, random_state=0)



scores = []
scores = cross_val_score(rf, df[features], df['shares'], cv=5, n_jobs=-1)


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
# print(accuracy_score(y_test,y_pred))
# print(scores)