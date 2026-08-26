from sklearn.linear_model import LinearRegression
X=[[30,20],[35,25],[40,30],[45,35],[50,40],
   [55,45],[60,50],[65,55],[70,60],[75,65]]
y=[8,10,13,17,22,28,35,43,52,62]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[58,48]]))
