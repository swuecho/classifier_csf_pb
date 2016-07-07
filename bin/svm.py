from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
from sklearn import svm


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
print(X_test)
clf = svm.SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
#pred vs y_test
print(pred)
print(y_test)
score = accuracy_score(y_test, pred)
print(score)

