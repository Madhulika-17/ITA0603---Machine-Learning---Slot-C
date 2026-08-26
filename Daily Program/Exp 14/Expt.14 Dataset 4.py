from sklearn.linear_model import LinearRegression
X=[[39.0,110,94],[38.5,108,95],[37.0,78,99],[39.2,115,93],
   [36.8,75,98],[38.8,112,94],[37.2,80,98],[39.1,114,92],
   [36.7,74,99],[38.9,111,93]]
y=[1,1,0,1,0,1,0,1,0,1]
m=LinearRegression()
m.fit(X,y)
p=m.predict([[38.7,109,94]])
print("Positive" if p[0]>=0.5 else "Negative")
