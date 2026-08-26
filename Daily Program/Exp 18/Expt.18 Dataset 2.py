from sklearn.linear_model import Perceptron
X=[[12,820,5],[10,790,6],[9,760,7],[7,700,8],[6,680,9],
   [5,650,10],[4,620,12],[8,730,8],[3,600,13],[11,810,5]]
y=[1,1,1,1,0,0,0,1,0,1]
m=Perceptron()
m.fit(X,y)
print(m.predict([[9,770,6]]))
