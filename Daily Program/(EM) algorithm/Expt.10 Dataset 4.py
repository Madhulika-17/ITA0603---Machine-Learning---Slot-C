from sklearn.mixture import GaussianMixture
import numpy as np
X=np.array([
[8,40000,0],
[7,35000,2],
[6,30000,4],
[5,25000,10],
[4,20000,20],
[9,45000,0],
[6,28000,5],
[5,23000,12],
[3,18000,25],
[8,42000,1]
])
g=GaussianMixture(n_components=4,random_state=0)
g.fit(X)
print(g.predict(X))
print("New Sample:",g.predict([[7,36000,3]]))
