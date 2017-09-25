'''
Name: Zhunissali Shanabek
Email: zshanabek@gmail.com
Date: 19.09.2017
Description: Code to check the decision tree classifier
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
dataset = load_iris()

X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=0)

dt = DecisionTreeClassifier(criterion="gini", splitter="best")

dt.fit(dataset.data, dataset.target)
predictedClass = dt.predict(X_test)

print(predictedClass)
#1. 
inst = [1.5, 2.2, 3.3,4.8]
print ('Decision Tree prediction:', dt.predict([inst]))
print ('Decision Tree prediction:', dt.predict_proba([inst]))
#2. 
print ('Accuracy: ', accuracy_score(Y_test, predictedClass))

