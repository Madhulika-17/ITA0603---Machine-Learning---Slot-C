from sklearn.linear_model import LinearRegression

X=[[5],[10],[15],[20],[25],[30],[35],[40],[45],[50]]
y=[1.2,2.5,4.1,6.2,8.8,11.9,15.5,19.6,24.2,29.3]

m=LinearRegression()
m.fit(X,y)

print(m.predict([[32]]))
