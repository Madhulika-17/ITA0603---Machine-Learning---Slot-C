from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

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

y = [1,1,0,1,0,1,0,1,0,1]

m = GaussianNB()
m.fit(X, y)

p = m.predict(X)

print("Prediction:", m.predict([[38.8,110,94]]))
print(confusion_matrix(y, p))
print("Accuracy:", accuracy_score(y, p))
