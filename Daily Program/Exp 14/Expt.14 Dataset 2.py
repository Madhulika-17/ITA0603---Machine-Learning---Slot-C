from sklearn.linear_model import LinearRegression
X=[[150,7.5,90],[160,7.8,88],[145,7.3,91],[120,5.5,70],
   [125,5.8,72],[130,6.0,74],[155,7.6,89],[118,5.4,69],
   [148,7.4,92],[122,5.7,71]]
y=[1,1,1,0,0,0,1,0,1,0]
m=LinearRegression()
m.fit(X,y)
p=m.predict([[152,7.5,90]])
print("Apple" if p[0]>=0.5 else "Orange")
