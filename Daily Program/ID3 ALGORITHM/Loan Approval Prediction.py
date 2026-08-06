from sklearn.tree import DecisionTreeClassifier

# Income: Low=0, Medium=1, High=2
# Credit Score: Poor=0, Average=1, Good=2
# Employment: Temporary=0, Permanent=1
# Property: No=0, Yes=1

X = [
    [2,2,1,1],
    [2,2,1,0],
    [1,2,1,1],
    [1,1,1,0],
    [0,0,0,0],
    [0,1,0,1],
    [2,1,1,1],
    [1,2,0,0],
    [2,2,1,1],
    [0,0,0,1]
]

y = [
    'Yes',
    'Yes',
    'Yes',
    'Yes',
    'No',
    'No',
    'Yes',
    'No',
    'Yes',
    'No'
]

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# New Sample: Medium, Good, Permanent, Yes
print(clf.predict([[1,2,1,1]]))
