from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
X=[[6.1,60,4000],[6.2,90,4500],[6.4,90,5000],[6.5,120,5000],[6.6,120,5500],
   [6.7,120,6000],[6.8,144,6000],[6.9,144,6500],[7.0,165,7000],[7.1,165,7000]]
y=[12000,15000,18000,23000,28000,34000,42000,52000,65000,78000]
print("DT:",DecisionTreeRegressor().fit(X,y).predict([[6.7,144,6000]]))
print("KNN:",KNeighborsRegressor().fit(X,y).predict([[6.7,144,6000]]))
