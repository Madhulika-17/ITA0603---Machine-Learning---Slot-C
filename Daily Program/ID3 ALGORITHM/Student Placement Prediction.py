from sklearn.tree import DecisionTreeClassifier

# CGPA: 5=0, 6=1, 7=2, 8=3, 9=4
# Communication: Poor=0, Average=1, Good=2, Excellent=3
# Internship: No=0, Yes=1
# Programming: Poor=0, Average=1, Good=2, Excellent=3

X = [
    [4,3,1,3],
    [3,2,1,2],
    [2,2,1,1],
    [1,1,0,1],
    [0,0,0,0],
    [4,3,1,2],
    [3,2,0,2],
    [1,1,1,1],
    [2,3,1,2],
    [0,0,0,1]
]

y = [
    'Placed',
    'Placed',
    'Placed',
    'Not Placed',
    'Not Placed',
    'Placed',
    'Placed',
    'Not Placed',
    'Placed',
    'Not Placed'
]

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# New Sample: 8, Excellent, Yes, Good
print(clf.predict([[3,3,1,2]]))
