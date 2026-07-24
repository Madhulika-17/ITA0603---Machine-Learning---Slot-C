from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

X = [
    [8,780,5],
    [7,760,4],
    [6,730,5],
    [4,620,8],
    [3,590,9],
    [9,810,4],
    [5,650,7],
    [8,770,5],
    [4,610,8],
    [7,750,6]
]

y = [1,1,1,0,0,1,0,1,0,1]

m = GaussianNB()
m.fit(X, y)

p = m.predict(X)

print("Prediction:", m.predict([[7,760,5]]))
print(confusion_matrix(y, p))
print("Accuracy:", accuracy_score(y, p))
