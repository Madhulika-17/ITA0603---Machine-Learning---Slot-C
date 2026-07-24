from sklearn.neighbors import KNeighborsClassifier

X = [
    [8,780,5],
    [7,760,4],
    [6,730,5],
    [4,620,8],
    [3,590,9],
    [9,810,4],
    [5,650,7],
    [8,770,5],
    [4,610,8],
    [7,750,6]
]

y = [
    'Yes',
    'Yes',
    'Yes',
    'No',
    'No',
    'Yes',
    'No',
    'Yes',
    'No',
    'Yes'
]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print("Prediction:", knn.predict([[7,750,5]]))
