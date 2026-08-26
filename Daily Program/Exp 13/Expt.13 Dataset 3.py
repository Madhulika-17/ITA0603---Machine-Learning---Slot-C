from sklearn.linear_model import LinearRegression
X=[[900,20,1],[1100,18,1],[1300,16,2],[1500,14,2],
   [1700,12,2],[1900,10,3],[2100,8,3],[2300,6,4],
   [2500,5,4],[2700,3,5]]
y=[40,48,60,74,88,104,122,142,165,190]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[1800,11,3]]))
