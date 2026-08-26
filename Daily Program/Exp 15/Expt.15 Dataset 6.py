from sklearn.naive_bayes import GaussianNB
X=[[100,95,70],[125,110,65],[150,125,55],[800,850,22],
   [1000,950,20],[1200,1050,18],[110,100,68],[900,900,21],
   [140,120,58],[1300,1100,17]]
y=['Bike','Bike','Bike','Car','Car','Car','Bike','Car','Bike','Car']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[135,118,60]]))
