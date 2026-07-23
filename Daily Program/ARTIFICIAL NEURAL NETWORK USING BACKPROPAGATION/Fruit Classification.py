from sklearn.neural_network import MLPClassifier

X = [
    [150,7.5,90],
    [160,7.8,88],
    [140,7.2,91],
    [120,5.5,70],
    [125,5.8,72],
    [130,6.0,74],
    [155,7.6,89],
    [118,5.4,69],
    [148,7.4,92],
    [122,5.7,71]
]

y = ['Apple','Apple','Apple','Orange','Orange',
     'Orange','Apple','Orange','Apple','Orange']

ann = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
ann.fit(X,y)

print("Prediction:", ann.predict([[152,7.5,90]]))
