import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas
import csv


layers = (5) #Attempt hyper tuning

with open('train.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

header = data[0]
print(header)
data = data[1:]

#clean data 2 - Name, 4 - Age
            
ages = []
for row in data:
    try:
        ages.append(float(row[4]))
    except:
        pass


meanage = np.mean(ages)

for row in data:
    if "master" in row[2].lower():
        row[2] = 1
    else:
        row[2] = 0
    if row[4] == None or row[4] == "":
        #row[4] = meanage
        del row

for i in range(len(data)):
    for j in range(len(header)):
        if type(data[i][j]) != type(3.2):
            if type(data[i][j]) == type(None):
                data[i][j] = 1
            if data[i][j] == "":
                data[i][j] = 0
            try:
                data[i][j] = float(data[i][j])
            except:
                print("-"+data[i][j]+"-")

#maxcol = [0 for i in range(len(header))]
#Try normalizing!
#for line in data:
#    for j in range(len(header)):
#        if line[j] > maxcol[j]:
#            maxcol[j] = line[j]
#print("maxcol", maxcol)
#for row in data:
#    for j in range(len(header)):
#        row[j] = row[j]/maxcol[j]


#print data
Y = [row[0] for row in data]
X = [row[1:] for row in data]

Ytrain = [Y[i] for i in range(len(Y)) if i%7 != 0]
Xtrain = [X[i] for i in range(len(Y)) if i%7 != 0]
Ytest = [Y[i] for i in range(len(Y)) if i%7 == 0]
Xtest = [X[i] for i in range(len(Y)) if i%7 == 0]


#X = [[1., 0.], [0., 0.],[0., 1.],[1.,1.]]
#y = [[1],[0],[1],[0]]



print("layers",layers)
sumloss = 0
for i in range(10):
    clf = MLPClassifier(solver='adam',max_iter=2000, hidden_layer_sizes=layers, random_state=i, tol=0.0000001)
    clf.fit(X, Y)
    print("test",clf.score(Xtest,Ytest))
    print("train",clf.score(Xtrain,Ytrain))
    sumloss += clf.score(Xtest,Ytest)
print("mean test score", sumloss/10)

clf = MLPClassifier(solver='adam',max_iter=2000, hidden_layer_sizes=layers, random_state=1, tol=0.0000001)
clf.fit(X, Y)
print("test",clf.score(Xtest,Ytest))
print("train",clf.score(Xtrain,Ytrain))
sumloss += clf.score(Xtest,Ytest)

#Final Submition Data
print("\n","final submition process")

with open('test.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

header = data[0]
print(header)
data = data[1:]

#clean data 2 - Name, 4 - Age
for row in data:
    if "master" in row[2].lower():
        row[1] = 1
    else:
        row[1] = 0
    if row[3] == None or row[3] == "":
        row[3] = meanage

for i in range(len(data)):
    for j in range(len(header)):
        if type(data[i][j]) != type(3.2):
            if type(data[i][j]) == type(None):
                data[i][j] = 1
            if data[i][j] == "":
                data[i][j] = 0
            try:
                data[i][j] = float(data[i][j])
            except:
                print("-"+data[i][j]+"-")

P = clf.predict(data)
P = [int(x) for x in P]
#Add ID
for i in range(len(P)):
    P[i] = [892+i,P[i]]

#Write to file
print("\nWrite?")
input()
with open('submit10.csv', mode='w', newline='') as f:
    submition = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    P = [["PassengerId", "Survived"]] + P
    for row in P:
        submition.writerow(row)
