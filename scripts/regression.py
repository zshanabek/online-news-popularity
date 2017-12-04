import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
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


logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X_train, y_train)

print ('Score RF:', logreg.score(X_test,y_test))


