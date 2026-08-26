from sklearn.linear_model import LinearRegression
X=[[5,150],[8,180],[10,220],[12,260],[15,300],
   [18,340],[20,380],[22,420],[25,460],[28,500]]
y=[220,260,310,360,420,490,570,650,740,840]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[18,350]]))
