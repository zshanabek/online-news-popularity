import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score
df = pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
    
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

dt = DecisionTreeClassifier(criterion="gini", splitter="best")
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
dt.fit(X_train, y_train)

predictedClassDT = dt.predict(X_test)
predictedClassKNN = knn.predict(X_test)

example_measures = np.array([[3,2,2,1,2,1,2,3,1]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = knn.predict(example_measures)

print ('Accuracy KNN:', knn.score(X_test,y_test))
print ('MSE KNN:', mean_squared_error(y_test, predictedClassKNN))
print ('KNN Prediction for {0} is {1}'.format(example_measures, prediction))


print ('Accuracy DT:', accuracy_score(y_test, predictedClassDT))
print ('MSE DT:', mean_squared_error(y_test, predictedClassDT))
