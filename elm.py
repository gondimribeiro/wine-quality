#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:20:22 2016

@author: george
"""
import numpy as np
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt

def preprocess_data():
    data = np.loadtxt(open("winequality-white.csv","rb"),delimiter=",",skiprows=1) 
    np.random.shuffle(data)
    
    X = data[:, 0:11]
    y_tmp = data[:, 11].astype(int)
    y_tmp.shape = [y_tmp.shape[0], 1]
    rows = np.arange(y_tmp.shape[0]).reshape(y_tmp.shape[0], 1)
    Y = np.zeros([y_tmp.shape[0], 10])
    Y[rows, y_tmp] = 1
    X = preprocessing.scale(X)
    X = X[:, [0,1,3,4,6,7,8,10]]
    
    i1 = int(math.ceil(0.7*X.shape[0]))
    i2 = int(math.ceil(0.15*X.shape[0]))
    
    X_train = X[0:i1, :]
    Y_train = Y[0:i1, :]
    
    X_val = X[i1:i1+i2, :]
    Y_val = Y[i1:i1+i2, :]

    X_test = X[i1+i2:, :]
    Y_test = Y[i1+i2:, :]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
def elm_activation(X, w_random):
    vtanh = np.vectorize(math.tanh)
    return vtanh(np.dot(X, w_random))    
        
def elm_optimize(H, N, n, s, C):
    if (n <= N):
        CI = np.multiply(np.identity(n+1), C)
        w = np.dot(H.transpose(), H)
        w = np.add(w, CI)
        w = np.linalg.inv(w)
        w = np.dot(w, H.transpose())
        w = np.dot(w, s)
    else:
        CI = np.multiply(np.identity(N), C)
        w = np.dot(H, H.transpose())
        w = np.add(w, CI)
        w = np.linalg.inv(w)
        w = np.dot(H.transpose(), w)
        w = np.dot(w, s)
    
    return w

X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data()
n = np.array([10, 30, 100, 300, 1000, 3000, 5000, 10000, 30000])
C = 2**np.array(range(-25, 25), dtype=np.float64)
C = np.concatenate(([0], C))

best_err_val = float('Inf')
err_train = []
err_val = []
init_mean = 0.3;

out_train = np.argmax(Y_train, axis=1)
out_val =  np.argmax(Y_val, axis=1)
out_test = np.argmax(Y_test, axis=1)
for nk in n:
    w_random = np.subtract(np.multiply(np.random.rand(X_train.shape[1], nk), init_mean*2), init_mean)
    
    # Training output
    H = elm_activation(X_train, w_random)
    ones = np.ones(H.shape[0])
    ones.shape = [H.shape[0], 1]
    H = np.concatenate((H, ones), axis=1)
    
    # Validation output    
    H_val = elm_activation(X_val, w_random)
    ones = np.ones(H_val.shape[0])
    ones.shape = [H_val.shape[0], 1]
    H_val = np.concatenate((H_val, ones), axis=1)
    
    # Test output
    H_test = elm_activation(X_test, w_random)
    ones = np.ones(H_test.shape[0])
    ones.shape = [H_test.shape[0], 1]
    H_test = np.concatenate((H_test, ones), axis=1)
    
    # C selection
    err_train.append([])
    err_val.append([])
    for Ck in C: 
        try:
            # Training
            w = elm_optimize(H, H.shape[0], nk, Y_train, Ck)
            diff = np.subtract(out_train, np.argmax(np.dot(H, w), axis=1))
            err_train[-1].append(np.multiply(diff, diff).sum() / float(diff.shape[0]))
            
            # Validation
            diff = np.subtract(out_val, np.argmax(np.dot(H_val, w), axis=1))
            err_val[-1].append(np.multiply(diff, diff).sum() / float(diff.shape[0]))
            
            if (err_val[-1][-1] < best_err_val):
                best_c = Ck
                best_n = nk
                best_w = w
                best_err_train = err_train[-1][-1]
                best_err_val = err_val[-1][-1]
                diff = np.subtract(out_test, np.argmax(np.dot(H_test, w), axis=1))
                err_test = np.multiply(diff, diff).sum() / float(diff.shape[0])
                print(best_n, best_c, np.linalg.norm(best_w), best_err_train, best_err_val, err_test)
        except np.linalg.LinAlgError:
                print("Exception: " + str(nk) + ", " + str(Ck))
  
                
print()
print(best_n, best_c, np.linalg.norm(best_w), best_err_train, best_err_val, err_test)
#legend1, = plt.plot(n, err_train)
#legend2, = plt.plot(n, err_val)    
#plt.legend([legend1, legend2], ['Treino', 'Validação'], loc='upper center', bbox_to_anchor=(0.5, 1.15),
#          ncol=3, fancybox=True, shadow=True)
#plt.xlabel('Número de neurônios')
#plt.ylabel('EQM')
#plt.show()