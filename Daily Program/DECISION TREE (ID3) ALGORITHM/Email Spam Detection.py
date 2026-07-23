from sklearn.tree import DecisionTreeClassifier

# Yes=1, No=0

X = [
    [1,1,1,0,1],
    [1,1,1,1,1],
    [0,0,0,0,0],
    [1,0,1,0,1],
    [0,1,0,1,0],
    [1,1,1,0,0],
    [1,0,1,1,1],
    [0,0,1,0,0],
    [1,1,0,0,1],
    [0,0,0,1,0]
]

y = [
    'Spam',
    'Spam',
    'Not Spam',
    'Spam',
    'Not Spam',
    'Spam',
    'Spam',
    'Not Spam',
    'Spam',
    'Not Spam'
]

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# New Sample: Yes Yes Yes No Yes
print(clf.predict([[1,1,1,0,1]]))
