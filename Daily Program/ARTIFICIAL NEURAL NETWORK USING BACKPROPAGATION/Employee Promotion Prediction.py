from sklearn.neural_network import MLPClassifier

X = [
    [10,95,50],
    [8,90,45],
    [7,88,40],
    [3,65,20],
    [2,60,18],
    [9,93,48],
    [4,70,25],
    [8,89,42],
    [2,58,15],
    [6,85,38]
]

y = ['Yes','Yes','Yes','No','No',
     'Yes','No','Yes','No','Yes']

ann = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
ann.fit(X,y)

print("Prediction:", ann.predict([[7,90,44]]))
