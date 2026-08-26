from sklearn.neighbors import KNeighborsRegressor
X=[[30,250,8],[35,300,8],[40,350,7],[45,400,7],
   [50,450,6],[55,500,6],[60,550,5],[65,600,5],
   [70,650,4],[75,700,4]]
y=[10.5,12.0,14.0,16.5,18.5,21.0,24.0,27.5,31.0,35.0]
m=KNeighborsRegressor()
m.fit(X,y)
print(m.predict([[52,470,6]]))
