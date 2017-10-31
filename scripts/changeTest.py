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
import xlsxwriter
csv_filename="OnlineNewsPopularity.csv"

df=pd.read_csv(csv_filename)

popular = df.shares >= 1400	
unpopular = df.shares < 1400

df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0

features=list(df.columns[2:60])
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], df['shares'], test_size=0.4, random_state=0)

# open one ouput excel file and two worksheets
workbook = xlsxwriter.Workbook('changeTest_output.xlsx')

worksheet = workbook.add_worksheet()
worksheet.write("A1","scala_index")
worksheet.write("B1","DecisionTree")
worksheet.write("C1","KNN")
worksheet.write("D1","RandomForest")
worksheet.write("E1","NaiveBayes")

worksheet2 = workbook.add_worksheet()
worksheet2.write("A1","scala_index")
worksheet2.write("B1","DecisionTree")
worksheet2.write("C1","KNN")
worksheet2.write("D1","RandomForest")
worksheet2.write("E1","NaiveBayes")

for i in range(0,100,5):
	X_rest, X_test_part, y_rest, y_test_part= model_selection.train_test_split(X_test, y_test, test_size=0.049+i/100.0, random_state=0)
	print "====================== loop: ", i 
	print "DecisionTree"
	dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
	clf_dt=dt.fit(X_train,y_train)
	score_dt=clf_dt.score(X_test_part,y_test_part)
	print "Acurracy: ", score_dt

	print "KNN"
	knn = KNeighborsClassifier()
	clf_knn=knn.fit(X_train, y_train)
	score_knn=clf_knn.score(X_test_part,y_test_part) 
	print "Acurracy: ", score_knn 

	print "RandomForest"
	rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
	clf_rf = rf.fit(X_train,y_train)
	score_rf=clf_rf.score(X_test_part,y_test_part)
	print "Acurracy: ", score_rf

	print "NaiveBayes"
	nb = BernoulliNB()
	clf_nb=nb.fit(X_train,y_train)
	score_nb=clf_nb.score(X_test_part,y_test_part)
	print "Acurracy: ", score_nb

	# write result data to excel file
	list1=[]
	list2=[]

	list1.append(i/100.0+0.05)
	list1.append(score_dt)
	list1.append(score_knn)
	list1.append(score_rf)
	list1.append(score_nb)

	for col in range(len(list1)):
		worksheet.write(i/5+1,col,list1[col])

workbook.close()
