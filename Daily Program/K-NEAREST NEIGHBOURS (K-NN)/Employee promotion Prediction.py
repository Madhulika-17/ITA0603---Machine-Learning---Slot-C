from sklearn.neighbors import KNeighborsClassifier

X = [
    [10,95,50],
    [9,92,48],
    [8,90,45],
    [4,70,25],
    [3,65,20],
    [11,96,52],
    [5,75,30],
    [8,89,44],
    [2,60,18],
    [7,87,40]
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

print("Prediction:", knn.predict([[7,90,44]]))
