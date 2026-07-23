from sklearn.neural_network import MLPClassifier

X = [
    [9.2,90,88],
    [8.8,85,82],
    [8.5,80,79],
    [7.2,72,70],
    [6.8,68,65],
    [9.0,92,90],
    [7.5,75,74],
    [8.6,84,81],
    [6.5,60,58],
    [8.9,88,86]
]

y = ['Placed','Placed','Placed','Not Placed','Not Placed',
     'Placed','Not Placed','Placed','Not Placed','Placed']

ann = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
ann.fit(X,y)

print("Prediction:", ann.predict([[8.7,86,84]]))
