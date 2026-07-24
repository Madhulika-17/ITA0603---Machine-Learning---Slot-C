from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

X = [
    [9.2,90,88],
    [8.8,85,84],
    [8.5,82,80],
    [7.2,72,70],
    [6.8,68,65],
    [9.0,91,89],
    [7.0,70,68],
    [8.7,86,83],
    [6.5,65,60],
    [9.1,92,90]
]

y = [1,1,1,0,0,1,0,1,0,1]

m = GaussianNB()
m.fit(X, y)

p = m.predict(X)

print("Prediction:", m.predict([[8.9,88,85]]))
print(confusion_matrix(y, p))
print("Accuracy:", accuracy_score(y, p))
