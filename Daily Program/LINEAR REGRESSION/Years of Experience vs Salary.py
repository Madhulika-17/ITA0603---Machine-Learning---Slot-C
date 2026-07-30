from sklearn.linear_model import LinearRegression

X=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y=[3.0,3.8,4.9,6.3,8.0,10.2,12.9,16.1,19.8,24.0]

m=LinearRegression()
m.fit(X,y)

print(m.predict([[7.5]]))
