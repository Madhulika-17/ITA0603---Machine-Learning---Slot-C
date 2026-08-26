from sklearn.mixture import GaussianMixture
import numpy as np
X=np.array([
[12,10,1,1],
[10,8,2,1],
[8,6,2,1],
[6,4,3,0],
[4,2,4,0],
[15,12,1,1],
[7,5,3,1],
[5,3,4,0],
[11,9,2,1],
[3,1,5,0]
])
g=GaussianMixture(n_components=4,random_state=0)
g.fit(X)
print(g.predict(X))
print("New Sample:",g.predict([[9,7,2,1]]))
