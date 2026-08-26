from sklearn.linear_model import Perceptron
X=[[15,92,6],[13,89,7],[11,85,8],[9,80,9],[7,72,10],
   [6,68,11],[5,65,12],[12,87,7],[4,60,13],[16,94,5]]
y=[1,1,1,1,0,0,0,1,0,1]
m=Perceptron()
m.fit(X,y)
print(m.predict([[12,88,7]]))
