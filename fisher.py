import numpy as np
import math
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def preprocess_data():
    data = np.loadtxt(open("winequality-white.csv","rb"),delimiter=",",skiprows=1) 
    np.random.shuffle(data)
    
    X = data[:, 0:11]
    Y = data[:, 11].astype(int)
    Y.shape = [Y.shape[0], 1]
    X = preprocessing.scale(X)
    X = X[:, [0,1,3,4,6,7,8,10]]
    
    i1 = int(math.ceil(0.7*X.shape[0]))
    i2 = int(math.ceil(0.15*X.shape[0]))
    
    X_train = np.array(X[0:i1+i2, :])
    Y_train = np.array(Y[0:i1+i2, :]).reshape(X_train.shape[0])
    
    X_test = np.array(X[i1+i2:, :])
    Y_test = np.array(Y[i1+i2:, :]).reshape(X_test.shape[0])

    return X_train, Y_train, X_test, Y_test

err_train = []
err_test  = []
for i in range(10):    
    X_train, Y_train, X_test, Y_test = preprocess_data()
    
    clf = LinearDiscriminantAnalysis(solver='lsqr')
    clf.fit(X_train, Y_train)
    
    diff = clf.predict(X_train) - Y_train
    err_train.append(np.multiply(diff, diff).sum() / float(diff.shape[0]))
    
    diff = clf.predict(X_test) - Y_test
    err_test.append(np.multiply(diff, diff).sum() / float(diff.shape[0]))


err_train = np.array(err_train)
err_test = np.array(err_test)
print(err_train.mean(), err_train.std())
print(err_test.mean(), err_test.std())

