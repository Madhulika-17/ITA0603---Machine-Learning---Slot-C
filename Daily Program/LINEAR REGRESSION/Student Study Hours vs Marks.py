from sklearn.linear_model import LinearRegression

X=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y=[18,30,43,57,68,78,86,92,96,98]

m=LinearRegression()
m.fit(X,y)

print(m.predict([[7.5]]))
