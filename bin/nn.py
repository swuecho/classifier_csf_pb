from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sknn.mlp import Classifier, Layer

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_test)

clf = Classifier(layers=[Layer("Rectifier", units=100),  Layer("Softmax")], learning_rate=0.02,n_iter=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

score = nn.score(X_test, y_test)
print(score)
print(y_test)
score = accuracy_score(y_test, y_pred)
print(score)

