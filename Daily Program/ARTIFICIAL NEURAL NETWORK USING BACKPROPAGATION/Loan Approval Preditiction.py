from sklearn.neural_network import MLPClassifier

X = [
    [8,780,5],
    [7,760,4],
    [6,720,6],
    [4,620,8],
    [3,580,10],
    [9,800,4],
    [5,650,7],
    [8,770,5],
    [4,600,9],
    [7,740,6]
]

y = ['Yes','Yes','Yes','No','No',
     'Yes','No','Yes','No','Yes']

ann = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
ann.fit(X,y)

print("Prediction:", ann.predict([[7,750,5]]))
