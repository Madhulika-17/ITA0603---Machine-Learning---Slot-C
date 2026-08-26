from sklearn.linear_model import LinearRegression
X=[[9.2,90,88],[8.8,85,82],[8.5,80,79],[7.2,72,70],
   [6.8,68,65],[9.0,92,90],[7.5,75,74],[8.6,84,81],
   [6.5,60,58],[8.9,88,86]]
y=[1,1,1,0,0,1,0,1,0,1]
m=LinearRegression()
m.fit(X,y)
p=m.predict([[8.7,86,84]])
print("Placed" if p[0]>=0.5 else "Not Placed")
