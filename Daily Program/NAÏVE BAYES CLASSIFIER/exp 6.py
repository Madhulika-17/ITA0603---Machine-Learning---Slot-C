from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

X = [[1], [2], [3], [6], [7], [8]]
y = [0, 0, 0, 1, 1, 1]

m = GaussianNB()
m.fit(X, y)

p = m.predict(X)

print(confusion_matrix(y, p))
print("Accuracy:", accuracy_score(y, p))
