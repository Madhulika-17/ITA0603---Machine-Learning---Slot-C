from sklearn.tree import DecisionTreeClassifier
X=[[39.0,110,94],[38.5,108,95],[37.0,78,99],[39.2,115,93],
   [36.8,75,98],[38.8,112,94],[37.2,80,98],[39.1,114,92],
   [36.7,74,99],[38.9,111,93]]
y=['Positive','Positive','Negative','Positive','Negative',
   'Positive','Negative','Positive','Negative','Positive']
m=DecisionTreeClassifier()
m.fit(X,y)
print(m.predict([[38.7,109,94]]))
