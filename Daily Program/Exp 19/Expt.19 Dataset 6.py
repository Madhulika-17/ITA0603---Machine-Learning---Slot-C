from sklearn.naive_bayes import GaussianNB
X=[[30,20],[35,25],[40,30],[45,35],[50,40],
   [55,45],[60,50],[65,55],[70,60],[75,65]]
y=['Low','Low','Low','Low','Low',
   'High','High','High','High','High']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[58,48]]))
