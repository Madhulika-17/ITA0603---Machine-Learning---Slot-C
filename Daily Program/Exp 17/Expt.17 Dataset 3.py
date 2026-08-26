from sklearn.tree import DecisionTreeClassifier
X=[[9.3,92,90],[8.9,87,85],[8.6,84,82],[7.3,74,72],[6.9,69,67],
   [9.1,91,89],[7.4,75,73],[8.7,85,83],[6.6,64,60],[8.8,88,86]]
y=['Placed','Placed','Placed','Not Placed','Not Placed','Placed',
   'Not Placed','Placed','Not Placed','Placed']
m=DecisionTreeClassifier()
m.fit(X,y)
print(m.predict([[8.8,86,84]]))
