# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:07:36 2019

@author: Haowei Huang
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import Ldata_helper as data_helper
import Lglobal_defs as global_defs
from sklearn import svm


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='linear', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict_KNN(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred
    def fit_predict_svm(self, Xs, Ys, Xt, Yt):
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = svm.SVC(C=1.0, kernel='rbf',gamma = 'scale', decision_function_shape='ovr')
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred, Xs_new, Xt_new
    def fit_predict_lin_svm(self, Xs_new, Ys, Xt_new, Yt):
        #Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = svm.LinearSVC()
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
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
        #D = [32, 64, 128, 256, 512, 1024, 2048]
        D = [2048]
        for dd in D:
            tca = TCA(dim=dd)
            acc2, ypre2, Xs_new, Xt_new = tca.fit_predict_svm(Xs, Ys, Xt, Yt)
            print(dd,"svm_rbf result:",acc2)
            acc3, ypre3 = tca.fit_predict_lin_svm(Xs_new, Ys, Xt_new, Yt)
            print(dd,"svm_lin result:",acc3)
        '''
        acc1, ypre1 = tca.fit_predict_KNN(Xs, Ys, Xt, Yt)
        print("1NN result:",acc1)
        '''
        '''
        acc3, ypre3 = tca.fit_predict_lin_svm(Xs, Ys, Xt, Yt)
        print("svm_lin result:",acc3)
        '''