import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score

df = pd.read_csv('OnlineNewsPopularity.csv')
df.drop(['url','timedelta'], 1, inplace=True)
for i in df['shares']:
    if int(i)>=1400:
        df['shares'].replace(i,1, inplace=True )   
    elif int(i)<1400 and int(i)!=1:
        df['shares'].replace(i, 0, inplace=True )           

# Transformation
x = df.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# Randomly select features
names = list(df.drop([58],1).columns.values)
dset = random.sample(names, 28)
dset1 = random.sample(names, 29)

X = np.array(df[dset])
X1 = np.array(df[dset1])
y = np.array(df[58])

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y,test_size=0.3,random_state=0)


knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(criterion="entropy", splitter="best")
gnb = GaussianNB()

knn1 = KNeighborsClassifier()
dt1 = DecisionTreeClassifier(criterion="entropy", splitter="best")
gnb1 = GaussianNB()

knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
gnb.fit(X_train, y_train)

knn1.fit(X_train1, y_train1)
dt1.fit(X_train1, y_train1)
gnb1.fit(X_train1, y_train1)


print ('Accuracy KNN:', knn.score(X_test,y_test))
print ('Accuracy DT:', dt.score(X_test,y_test))
print ('Accuracy GNB:', gnb.score(X_test,y_test))

print ('Accuracy KNN1:', knn1.score(X_test1,y_test1))
print ('Accuracy DT1:', dt1.score(X_test1,y_test1))
print ('Accuracy GNB1:', gnb1.score(X_test1,y_test1))


