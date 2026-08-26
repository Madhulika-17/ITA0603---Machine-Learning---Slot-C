from sklearn.naive_bayes import GaussianNB
X=[[1000,120],[1200,150],[1400,180],[1600,220],[1800,260],
   [2000,300],[2200,340],[2400,380],[2600,420],[2800,460]]
y=['Low','Low','Low','Low','Low',
   'High','High','High','High','High']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[2100,320]]))
