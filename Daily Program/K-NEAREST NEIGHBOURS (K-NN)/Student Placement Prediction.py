from sklearn.neighbors import KNeighborsClassifier

X = [
    [9.1,90,88],
    [8.8,85,84],
    [8.2,82,80],
    [7.0,70,68],
    [6.5,65,60],
    [9.3,92,90],
    [7.4,72,70],
    [8.7,86,83],
    [6.8,67,65],
    [9.0,89,87]
]

y = [
    'Placed',
    'Placed',
    'Placed',
    'Not Placed',
    'Not Placed',
    'Placed',
    'Not Placed',
    'Placed',
    'Not Placed',
    'Placed'
]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print("Prediction:", knn.predict([[8.6,86,84]]))
