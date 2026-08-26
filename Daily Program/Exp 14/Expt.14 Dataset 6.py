from sklearn.linear_model import LinearRegression
X=[[100,95,70],[125,110,65],[150,125,55],[800,850,22],
   [1000,950,20],[1200,1050,18],[110,100,68],[900,900,21],
   [140,120,58],[1300,1100,17]]
y=[1,1,1,0,0,0,1,0,1,0]
m=LinearRegression()
m.fit(X,y)
p=m.predict([[135,118,60]])
print("Bike" if p[0]>=0.5 else "Car")
