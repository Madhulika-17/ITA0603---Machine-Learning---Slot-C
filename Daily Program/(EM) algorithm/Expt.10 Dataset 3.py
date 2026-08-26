from sklearn.mixture import GaussianMixture
import numpy as np
X=np.array([
[90000,45,5,0],
[75000,40,6,0],
[60000,35,8,0],
[45000,30,10,1],
[30000,25,12,1],
[95000,48,4,0],
[65000,37,7,0],
[40000,28,9,1],
[28000,24,13,1],
[85000,43,5,0]
])
g=GaussianMixture(n_components=4,random_state=0)
g.fit(X)
print(g.predict(X))
print("New Sample:",g.predict([[70000,38,6,0]]))
