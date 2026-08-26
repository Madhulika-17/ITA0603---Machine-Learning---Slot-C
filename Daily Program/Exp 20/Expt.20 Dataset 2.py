from sklearn.linear_model import LinearRegression
X=[[20,2],[25,2],[30,3],[35,3],[40,4],
   [45,4],[50,5],[55,5],[60,6],[65,6]]
y=[250,290,340,390,450,510,580,640,710,780]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[48,5]]))
