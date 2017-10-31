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



csv_filename="OnlineNewsPopularity.csv"

df=pd.read_csv(csv_filename)

popular = df.shares >= 1400	
unpopular = df.shares < 1400

df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0


features=list(df.columns[2:60])
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], df['shares'], test_size=0.4, random_state=0)

print "RandomForest"
rf1 = RandomForestClassifier(n_estimators=100,n_jobs=-1, criterion="entropy")
clf_rf1 = rf1.fit(X_train,y_train)
score_rf1=clf_rf1.score(X_test,y_test)
print "Acurracy: ", score_rf1

# increasingly add size of training set 5% of orginal, keep testing size unchanged
for i in range(0,100,5):
	X_rest, X_trian_part, y_rest, y_train_part= model_selection.train_test_split(X_train, y_train, test_size=0.049+i/100.0, random_state=0)
	print "====================== loop: ", i 
	t0=time()
	print "DecisionTree"
	dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
# dt = DecisionTreeClassifier(min_samples_split=20,max_depth=5,random_state=99)
	clf_dt=dt.fit(X_trian_part,y_train_part)
	score_dt=clf_dt.score(X_test,y_test)
	print "Acurracy: ", score_dt
	t1=time()
	dur_dt=t1-t0
	print "time elapsed: ", dur_dt	
	print "\n"

	t6=time()
	print "KNN"
# knn = KNeighborsClassifier(n_neighbors=3)
	knn = KNeighborsClassifier()
	clf_knn=knn.fit(X_trian_part, y_train_part)
	score_knn=clf_knn.score(X_test,y_test) 
	print "Acurracy: ", score_knn 
	t7=time()
	dur_knn=t7-t6
	print "time elapsed: ", dur_knn
	print "\n"

	t2=time()
	print "RandomForest"
	rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
	clf_rf = rf.fit(X_trian_part,y_train_part)
	score_rf=clf_rf.score(X_test,y_test)
	print "Acurracy: ", score_rf
	t3=time()
	dur_rf=t3-t2
	print "time elapsed: ", dur_rf
	print "\n"

	t4=time()
	print "NaiveBayes"
	nb = BernoulliNB()
	clf_nb=nb.fit(X_trian_part,y_train_part)
	score_nb=clf_nb.score(X_test,y_test)
	print "Acurracy: ", score_nb
	t5=time()
	dur_nb=t5-t4
	print "time elapsed: ", dur_nb
