from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
X=[[4,6,64],[6,8,128],[8,8,128],[8,8,256],[12,8,256],
   [12,8,512],[16,8,512],[16,8,1024],[18,8,1024],[24,8,1024]]
y=[14000,19000,26000,32000,42000,52000,65000,82000,95000,120000]
print("DT:",DecisionTreeRegressor().fit(X,y).predict([[12,8,512]]))
print("KNN:",KNeighborsRegressor().fit(X,y).predict([[12,8,512]]))
