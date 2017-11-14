import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score,f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_auc_score
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

rf = RandomForestClassifier(n_estimators=100, criterion="entropy")
clf_rf = rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

predictedClassRF = clf_rf.predict(X_test)
print ('Score RF:', clf_rf.score(X_test,y_test))
print ('MSE RF:', mean_squared_error(y_test, predictedClassRF))
print ('Accuracy:', accuracy_score(y_test, y_pred))
print ('Precision Score:', precision_score(y_test, y_pred))
print ('Recall Score:', recall_score(y_test, y_pred, average='macro'))
print ('ROC AUC Score:', roc_auc_score(y_test, y_pred))
print ('F1 Score:', f1_score(y_test, y_pred, average='macro'))

