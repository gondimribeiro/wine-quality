import numpy as np
import math
from sklearn import preprocessing
from sklearn import svm

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
    
    X_train = X[0:i1, :]
    Y_train = Y[0:i1, :].reshape(X_train.shape[0])
    
    X_val = X[i1:i1+i2, :]
    Y_val = Y[i1:i1+i2, :].reshape(X_val.shape[0])
    
    X_test = X[i1+i2:, :]
    Y_test = Y[i1+i2:, :].reshape(X_test.shape[0])
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data()
C = 2**np.array(range(-25, 25), dtype=np.float64)
gamma = C
best_err = float('Inf')

for Ck in C:
    for gk in gamma:
        clf = svm.SVC(kernel='rbf', C=Ck, gamma=gk)
        clf.fit(X_train, Y_train)
        
        err_train=[]
        diff = clf.predict(X_train) - Y_train
        err_train = np.multiply(diff, diff).sum() / float(diff.shape[0])
        
        diff = clf.predict(X_val) - Y_val
        err_val = np.multiply(diff, diff).sum() / float(diff.shape[0])
        
        if err_val < best_err:
            best_c = Ck
            best_g = gk
            best_svm = clf
            best_val = err_val
            best_err = err_val
            diff = clf.predict(X_test) - Y_test
            err_test = np.multiply(diff, diff).sum() / float(diff.shape[0])
            print(Ck, gk, err_train, err_val, err_test)                        
