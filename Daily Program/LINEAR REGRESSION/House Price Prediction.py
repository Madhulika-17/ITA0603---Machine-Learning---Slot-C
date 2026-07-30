from sklearn.linear_model import LinearRegression

X=[[800],[1000],[1200],[1400],[1600],[1800],[2000],[2200],[2400],[2600]]
y=[32,40,52,68,88,112,140,172,208,248]

m=LinearRegression()
m.fit(X,y)

print(m.predict([[1700]]))
