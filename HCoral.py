# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:24:35 2019

@author: Haowei Huang
"""
import numpy as np
#import scipy.io
import scipy.linalg
import Ldata_helper as data_helper
import Lglobal_defs as global_defs
import sklearn.metrics
import sklearn.neighbors
from sklearn import svm


class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new

    def fit_predict_KNN(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred
    def fit_predict_svm(self, Xs, Ys, Xt, Yt):
        Xs_new = self.fit(Xs, Xt)
        print(Xs_new.shape)
        clf = svm.SVC(C=5.0, kernel='rbf',gamma = 'scale', decision_function_shape='ovr')
        clf.fit(Xs_new, Ys.ravel())#.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred
    def fit_predict_lin_svm(self, Xs, Ys, Xt, Yt):
        Xs_new = self.fit(Xs, Xt)
        clf = svm.LinearSVC()
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


if __name__ == '__main__':
    for i in range(0,3):
        [src,dst] = data_helper.read_paired_labeled_features(i)
        Xs = src[0]
        Ys = src[1]
        Xt = dst[0]
        Yt = dst[1]
        print(Xs.shape)
        print(Ys.shape)
        coral = CORAL()
        '''
        acc1, ypre1 = coral.fit_predict_KNN(Xs, Ys, Xt, Yt)
        print("1NN result:",acc1)
        '''
        acc2, ypre2 = coral.fit_predict_svm(Xs, Ys, Xt, Yt)
        print("svm_rbf result:",acc2)
        '''
        acc3, ypre3 = coral.fit_predict_lin_svm(Xs, Ys, Xt, Yt)
        print("svm_lin result:",acc3)
        '''
