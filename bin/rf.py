from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


def vectorize(cdr3):
    v =np.zeros(50) 
    for i in range(len(cdr3)):
        v[i] = ord(cdr3[i]) - 64
    return(v)

dataset = pd.read_csv("../input/dataset.csv")
y= dataset['subset'].ravel()
X= dataset['cdr3'].ravel()

for i in range(len(X)):
    X[i] = vectorize(X[i])
X=np.array([np.array(xi) for xi in X])
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train)
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
#pred vs y_test
print(pred)
print(y_test)
for i in zip(y_test,pred):
    if i[0] != i[1]:
        print(i)

x = (sum(list(y_test))/len(y_test))
print(( 1 - x))
score = accuracy_score(y_test, pred)
print(score)

