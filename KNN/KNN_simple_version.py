import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


###load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataSet = pandas.read_csv(url, names=names)

'''
>>> dataset['class'].value_counts()
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
'''

##get testset
testSet = [6.5 , 3.0 , 5.2 , 2.0]


##calculate distance
diff=dataset.copy()
diff['distance']=0
label=names[len(names)-1]

for i in range(len(names)-1):
    col=names[i]
    diff[col]=diff[col]-testSet[i]
    diff['distance']=diff['distance']+pow(diff[col],2)

diff['distance']=pow(diff['distance'],0.5)    

##sort by distance
topK=diff.sort_values(by=["distance"],ascending=True).head(10)

##get result
itemClass=topK[label].value_counts().reset_index().sort_values(by=["index"],ascending=False)
result=itemClass.ix[0,0]


def kNNClassify(testSet,dataSet,k):
    diff=dataSet.copy()
    diff['distance']=0
    label=names[len(names)-1]
    
    for i in range(len(names)-1):
        col=names[i]
        diff[col]=diff[col]-testSet[i]
        diff['distance']=diff['distance']+pow(diff[col],2)

    diff['distance']=pow(diff['distance'],0.5)    

    ##sort by distance
    topK=diff.sort_values(by=["distance"],ascending=True).head(k)

    ##get result
    itemClass=topK[label].value_counts().reset_index().sort_values(by=["index"],ascending=False)
    return itemClass.ix[0,0]

testSet = [[6.5 , 3.0 , 5.2 , 2.0],[6.5 , 3.0 , 5.2 , 2.0]]
for i in testSet:
    print kNNClassify(i,dataset,10)
    
