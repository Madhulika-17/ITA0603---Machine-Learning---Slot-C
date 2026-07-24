from sklearn.neighbors import KNeighborsClassifier

X = [
    [39.1,112,94],
    [38.8,109,95],
    [37.0,78,99],
    [39.3,115,93],
    [36.8,74,98],
    [38.7,110,94],
    [37.1,80,98],
    [39.0,113,92],
    [36.9,76,99],
    [38.9,111,93]
]

y = [
    'Positive',
    'Positive',
    'Negative',
    'Positive',
    'Negative',
    'Positive',
    'Negative',
    'Positive',
    'Negative',
    'Positive'
]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print("Prediction:", knn.predict([[38.8,110,94]]))
