from sklearn.linear_model import Perceptron
X=[[20,10,1],[18,9,1],[15,8,2],[12,6,2],[10,5,3],
   [8,4,3],[6,3,4],[16,8,2],[5,2,5],[22,12,1]]
y=[1,1,1,1,0,0,0,1,0,1]
m=Perceptron()
m.fit(X,y)
print(m.predict([[17,8,2]]))
