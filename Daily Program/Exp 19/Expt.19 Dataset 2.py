from sklearn.naive_bayes import GaussianNB
X=[[20,2],[25,2],[30,3],[35,3],[40,4],
   [45,4],[50,5],[55,5],[60,6],[65,6]]
y=['Low','Low','Low','Low','High',
   'High','High','High','High','High']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[48,5]]))
