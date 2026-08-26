from sklearn.linear_model import LinearRegression
X=[[800,2,15],[1000,2,12],[1200,3,10],[1400,3,8],
   [1600,4,6],[1800,4,5],[2000,5,4],[2200,5,3],
   [2400,6,2],[2600,6,1]]
y=[35,42,52,63,75,88,102,118,135,150]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[1700,4,5]]))
