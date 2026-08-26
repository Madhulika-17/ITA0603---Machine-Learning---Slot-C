from sklearn.linear_model import LinearRegression
X=[[700,1,15],[850,2,12],[1000,3,10],[1150,4,8],
   [1300,5,7],[1450,6,5],[1600,7,4],[1750,8,3],
   [1900,9,2],[2050,10,1]]
y=[30,38,48,60,72,86,100,116,134,154]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[1500,6,4]]))
