from sklearn.tree import DecisionTreeClassifier

# Yes=1, No=0

X = [
    [1,1,1,1,1],
    [1,1,0,1,1],
    [0,1,1,0,0],
    [1,0,1,1,1],
    [0,0,0,0,0],
    [1,1,1,0,1],
    [0,1,0,1,0],
    [1,0,0,1,1],
    [1,1,1,1,1],
    [0,0,1,0,0]
]

y = [
    'Positive',
    'Positive',
    'Negative',
    'Positive',
    'Negative',
    'Positive',
    'Negative',
    'Positive',
    'Positive',
    'Negative'
]

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# New Sample: Yes Yes No Yes Yes
print(clf.predict([[1,1,0,1,1]]))
