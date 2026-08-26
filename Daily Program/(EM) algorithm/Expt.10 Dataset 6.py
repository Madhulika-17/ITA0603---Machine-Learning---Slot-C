from sklearn.mixture import GaussianMixture
import numpy as np
X=np.array([
[15,850,1,1],
[12,800,2,1],
[10,760,2,1],
[8,700,3,2],
[6,620,4,0],
[16,880,1,1],
[11,780,2,1],
[7,680,3,2],
[5,600,5,0],
[14,830,1,1]
])
g=GaussianMixture(n_components=4,random_state=0)
g.fit(X)
print(g.predict(X))
print("New Sample:",g.predict([[11,790,2,1]]))
