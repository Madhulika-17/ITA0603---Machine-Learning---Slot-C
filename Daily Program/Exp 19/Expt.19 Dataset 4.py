from sklearn.naive_bayes import GaussianNB
X=[[1000,5000],[1500,7000],[2000,9000],[2500,11000],[3000,13000],
   [3500,15000],[4000,17000],[4500,19000],[5000,21000],[5500,23000]]
y=['Low','Low','Low','Low','Low',
   'High','High','High','High','High']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[4200,18000]]))
