from sklearn.linear_model import LinearRegression
X=[[10,95,50],[8,90,45],[7,88,40],[3,65,20],
   [2,60,18],[9,93,48],[4,70,25],[8,89,42],
   [2,58,15],[6,85,38]]
y=[1,1,1,0,0,1,0,1,0,1]
m=LinearRegression()
m.fit(X,y)
p=m.predict([[7,90,44]])
print("Promoted" if p[0]>=0.5 else "Not Promoted")
