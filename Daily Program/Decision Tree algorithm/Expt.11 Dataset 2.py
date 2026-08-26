from sklearn.tree import DecisionTreeClassifier
X=[[150,7.5,90],[160,7.8,88],[145,7.3,91],[120,5.5,70],
   [125,5.8,72],[130,6.0,74],[155,7.6,89],[118,5.4,69],
   [148,7.4,92],[122,5.7,71]]
y=['Apple','Apple','Apple','Orange','Orange','Orange',
   'Apple','Orange','Apple','Orange']
m=DecisionTreeClassifier()
m.fit(X,y)
print(m.predict([[152,7.5,90]]))
