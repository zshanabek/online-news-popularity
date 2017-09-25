from sklearn.neighbors import KNeighborsClassifier
f1 = [
    [0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,0,5],[0,0,6],[0,0,7],[0,0,8],[0,0,9],
    [1,0,0],[1,0,1]
    ]
y = [0,1,0,1,0,1,0,1,0,1,0,1]
knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(f1,y)

print(knn.predict([[0.2,0.2,6.4]]))
print(knn.predict_proba([[0.2,0.2,6.4]]))
