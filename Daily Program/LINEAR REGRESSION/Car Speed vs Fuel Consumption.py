from sklearn.linear_model import LinearRegression

X=[[40],[50],[60],[70],[80],[90],[100],[110],[120],[130]]
y=[4.5,4.8,5.2,5.9,6.8,8.0,9.5,11.2,13.4,16.0]

m=LinearRegression()
m.fit(X,y)

print(m.predict([[95]]))
