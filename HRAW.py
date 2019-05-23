# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:03:28 2019

@author: 12709
"""
import numpy as np
#import scipy.io
import scipy.linalg
import Ldata_helper as data_helper
import Lglobal_defs as global_defs
import sklearn.metrics
import sklearn.neighbors
from sklearn import svm
if __name__ == '__main__':
    for i in range(0,3):
        [src,dst] = data_helper.read_paired_labeled_features(i)
        Xs = src[0]
        Ys = src[1]
        Xt = dst[0]
        Yt = dst[1]
        print(len(np.unique(Ys)))
        print(np.unique(Ys))
        print(Xt.shape)
        '''
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        print('1NN',acc)
        clf = svm.LinearSVC()
        clf.fit(Xs, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        print('svm_lin',acc)
        '''
        '''
        clf = svm.SVC(C=1.0, kernel='rbf',gamma = 'scale', decision_function_shape='ovr')
        clf.fit(Xs, Ys.ravel())#.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        print('svm_rbf',acc)
        '''