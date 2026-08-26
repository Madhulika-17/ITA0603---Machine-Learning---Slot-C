from sklearn.naive_bayes import GaussianNB
X=[[5,150],[8,180],[10,220],[12,260],[15,300],
   [18,340],[20,380],[22,420],[25,460],[28,500]]
y=['Low','Low','Low','Low','Low',
   'High','High','High','High','High']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[18,350]]))
