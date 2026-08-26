from sklearn.naive_bayes import GaussianNB
X=[[8,780,5],[7,760,4],[6,730,5],[4,620,8],[3,590,9],
   [9,810,4],[5,650,7],[8,770,5],[4,610,8],[7,750,6]]
y=['Approved','Approved','Approved','Rejected','Rejected',
   'Approved','Rejected','Approved','Rejected','Approved']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[7,760,5]]))
