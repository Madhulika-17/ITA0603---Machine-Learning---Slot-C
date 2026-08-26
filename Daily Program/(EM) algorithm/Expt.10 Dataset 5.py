from sklearn.mixture import GaussianMixture
import numpy as np
X=np.array([
[10,30000,1],
[8,35000,2],
[6,40000,3],
[4,45000,5],
[2,50000,7],
[11,28000,1],
[7,36000,2],
[5,43000,4],
[3,48000,6],
[9,32000,1]
])
g=GaussianMixture(n_components=4,random_state=0)
g.fit(X)
print(g.predict(X))
print("New Sample:",g.predict([[7,34000,2]]))
