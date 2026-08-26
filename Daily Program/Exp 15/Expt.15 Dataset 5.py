from sklearn.naive_bayes import GaussianNB
X=[[10,95,50],[8,90,45],[7,88,40],[3,65,20],[2,60,18],
   [9,93,48],[4,70,25],[8,89,42],[2,58,15],[6,85,38]]
y=['Promoted','Promoted','Promoted','Not Promoted','Not Promoted',
   'Promoted','Not Promoted','Promoted','Not Promoted','Promoted']
m=GaussianNB()
m.fit(X,y)
print(m.predict([[7,90,44]]))
