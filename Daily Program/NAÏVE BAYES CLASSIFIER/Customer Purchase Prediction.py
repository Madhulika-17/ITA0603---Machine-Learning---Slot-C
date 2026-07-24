from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

X = [
    [22,3,30],
    [25,4,35],
    [28,5,45],
    [32,7,60],
    [35,8,65],
    [40,9,75],
    [45,10,80],
    [30,6,55],
    [24,4,40],
    [38,9,70]
]

y = [0,0,0,1,1,1,1,1,0,1]

m = GaussianNB()
m.fit(X, y)

p = m.predict(X)

print("Prediction:", m.predict([[34,8,68]]))
print(confusion_matrix(y, p))
print("Accuracy:", accuracy_score(y, p))
