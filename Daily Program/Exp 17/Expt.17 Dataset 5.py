from sklearn.tree import DecisionTreeClassifier
X=[[10,95,50],[9,92,48],[8,90,45],[4,72,28],[3,68,24],
   [11,96,52],[5,75,30],[8,89,44],[2,60,18],[7,87,40]]
y=['Promoted','Promoted','Promoted','Not Promoted','Not Promoted',
   'Promoted','Not Promoted','Promoted','Not Promoted','Promoted']
m=DecisionTreeClassifier()
m.fit(X,y)
print(m.predict([[8,90,46]]))
