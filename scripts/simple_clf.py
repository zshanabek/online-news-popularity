import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score,f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_auc_score, roc_curve
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
y_pred_rf = rf.predict(X_test)
predictedClassRF = clf_rf.predict(X_test)

dt = DecisionTreeClassifier(criterion="gini", splitter="random")
clf_dt = dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
predictedClassDT = clf_dt.predict(X_test)

knn = KNeighborsClassifier()
clf_knn = knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
predictedClassKNN = clf_knn.predict(X_test)

nb = BernoulliNB()
clf_nb = nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
predictedClassNB = clf_nb.predict(X_test)

lr = LogisticRegression()
clf_lr = lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
predictedClassLR = clf_lr.predict(X_test)

print ("============Random Forest============")
print ('MSE RF:', round(mean_squared_error(y_test, predictedClassRF), 4))
print ('Accuracy:', round(accuracy_score(y_test, y_pred_rf), 4))
print ('Precision Score:', round(precision_score(y_test, y_pred_rf), 4))
print ('Recall Score:', round(recall_score(y_test, y_pred_rf, average='macro'), 4))
print ('ROC AUC Score:', round(roc_auc_score(y_test, y_pred_rf), 4))
print ('F1 Score:', round(f1_score(y_test, y_pred_rf, average='macro'), 4))

print ("============Decision Tree============")
print ('MSE:', round(mean_squared_error(y_test, predictedClassDT), 4))
print ('Accuracy:', round(accuracy_score(y_test, y_pred_dt), 4))
print ('Precision Score:', round(precision_score(y_test, y_pred_dt), 4))
print ('Recall Score:', round(recall_score(y_test, y_pred_dt, average='macro'), 4))
print ('ROC AUC Score:', round(roc_auc_score(y_test, y_pred_dt), 4))
print ('F1 Score:', round(f1_score(y_test, y_pred_dt, average='macro'), 4))

print ("============KNN============")
print ('MSE:', round(mean_squared_error(y_test, predictedClassKNN), 4))
print ('Accuracy:', round(accuracy_score(y_test, y_pred_knn), 4))
print ('Precision Score:',round( precision_score(y_test, y_pred_knn)))
print ('Recall Score:', round(recall_score(y_test, y_pred_knn, average='macro'), 4))
print ('ROC AUC Score:', round(roc_auc_score(y_test, y_pred_knn), 4))
print ('F1 Score:', round(f1_score(y_test, y_pred_knn, average='macro'), 4))

print ("============Logistic Regression============")
print ('MSE:', round(mean_squared_error(y_test, predictedClassLR), 4))
print ('Accuracy:', round(accuracy_score(y_test, y_pred_lr), 4))
print ('Precision Score:', round(precision_score(y_test, y_pred_lr), 4))
print ('Recall Score:', round(recall_score(y_test, y_pred_lr, average='macro'), 4))
print ('ROC AUC Score:', round(roc_auc_score(y_test, y_pred_lr), 4))
print ('F1 Score:', round(f1_score(y_test, y_pred_lr, average='macro'), 4))