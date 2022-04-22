import pandas as pd
import numpy as np
import re

def tf(corpus,Vocabulary):
    matrix=np.zeros((len(corpus),len(Vocabulary)))
    for c in range(len(corpus)):
        for v in range(len(Vocabulary)):
            matrix[c][v]=corpus[c].count(Vocabulary[v])
    return matrix

def dfx(corpus,Vocabulary):
    cpt=np.zeros(len(Vocabulary))
    for v in range(len(Vocabulary)):
        for c in corpus:
            if Vocabulary[v] in c:
                cpt[v]+=1
    return cpt
def tf_idfx(X_train,X_test,Vocabulary):
    dfx_train=dfx(X_train,Vocabulary)
    dfx_test=dfx(X_test,Vocabulary)
    tfx_train=tf(X_train,Vocabulary)
    tfx_test=tf(X_test,Vocabulary)
    N_train=tfx_train.shape[0]
    N_test=tfx_test.shape[0] 
    tf_idf_train=tfx_train*np.log(N_train/dfx_train)
    tf_idf_test=tfx_test*np.log(N_test/dfx_test)
    return  tf_idf_train,tf_idf_test