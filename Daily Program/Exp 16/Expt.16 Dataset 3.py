from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
X=[[1.8,4,4000],[2.0,4,4500],[2.2,6,5000],[2.4,6,5000],[2.6,8,5500],
   [2.8,8,6000],[3.0,12,6000],[3.2,12,6500],[3.4,16,7000],[3.6,16,7000]]
y=[10000,12000,17000,21000,26000,32000,40000,50000,62000,76000]
print("DT:",DecisionTreeRegressor().fit(X,y).predict([[2.9,8,6000]]))
print("KNN:",KNeighborsRegressor().fit(X,y).predict([[2.9,8,6000]]))
