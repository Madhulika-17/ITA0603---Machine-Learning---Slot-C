from sklearn.tree import DecisionTreeClassifier
X=[[39.2,112,93],[38.8,110,94],[37.0,78,99],[39.1,114,92],
   [36.8,76,98],[38.9,111,94],[37.2,80,99],[39.3,115,92],
   [36.7,74,98],[38.7,109,95]]
y=['Positive','Positive','Negative','Positive','Negative','Positive',
   'Negative','Positive','Negative','Positive']
m=DecisionTreeClassifier()
m.fit(X,y)
print(m.predict([[38.9,110,94]]))
