from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

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

y = [1,1,1,0,0,1,0,1,0,1]

m = GaussianNB()
m.fit(X, y)

p = m.predict(X)

print("Prediction:", m.predict([[8,91,46]]))
print(confusion_matrix(y, p))
print("Accuracy:", accuracy_score(y, p))
