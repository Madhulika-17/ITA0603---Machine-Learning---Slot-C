from sklearn.linear_model import LinearRegression
X=[[1000,120],[1200,150],[1400,180],[1600,220],[1800,260],
   [2000,300],[2200,340],[2400,380],[2600,420],[2800,460]]
y=[8,10,12,15,18,22,26,31,36,42]
m=LinearRegression()
m.fit(X,y)
print(m.predict([[2100,320]]))
