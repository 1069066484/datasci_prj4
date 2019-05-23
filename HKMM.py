# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:29:48 2019

@author: 12709
"""

import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers
import Ldata_helper as data_helper
import Lglobal_defs as global_defs
from sklearn import svm

def main_example():
    [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(global_defs.DA.P2R)
    src_dl = data_helper.shuffle_labeled_data(src_dl)

    # First, test your model without domain adaptation
    clf = svm.LinearSVC()
    clf.fit(src_dl[0], src_dl[1].ravel(), max_iter=100)
    y_pred = clf.predict(tgt_dl[0])
    acc_without_da = sklearn.metrics.accuracy_score(tgt_dl[1], y_pred)

    # You need to split the target dataset into training set and test set
    tgt_tr_dl, tgt_te_dl = data_helper.labeled_data_split(tgt_dl, 0.6)
    kmm = KMM(kernel_type='rbf', B=10)
    beta = kmm.fit(Xs, Xt)
    acc_training_with_da, _, clf = kmm.fit_predict_svm(src_dl[0], src_dl[1], tgt_tr_dl[0], tgt_tr_dl[1], beta)
    y_pred = clf.predict(tgt_te_dl[0])
    acc_test_with_da = sklearn.metrics.accuracy_score(tgt_te_dl[1], y_pred)
    print("Accuracy without domain adaptation = ", acc_without_da)
    print("Training accuracy with domain adaptation = ", acc_training_with_da)
    print("Test accuracy with domain adaptation = ", acc_test_with_da)

def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K

class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta
    def fit_predict_svm(self, Xs, Ys, Xt, Yt, beta):
        weight = beta
        clf = svm.SVC(C=1.0, kernel='rbf',gamma = 'scale', decision_function_shape='ovr')
        clf.fit(Xs, Ys.ravel(), weight.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred
    def fit_predict_lin_svm(self, Xs, Ys, Xt, Yt, beta):
        weight = beta
        clf = svm.LinearSVC()
        clf.fit(Xs, Ys.ravel(), weight.ravel())
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
        kmm = KMM(kernel_type='linear', B=10)
        beta = kmm.fit(Xs, Xt)
        print(beta)
        print(beta.shape)
        acc2, ypre2 = kmm.fit_predict_svm(Xs, Ys, Xt, Yt, beta)
        print("svm_rbf result:",acc2)
        acc3, ypre3 = kmm.fit_predict_lin_svm(Xs, Ys, Xt, Yt, beta)
        print("svm_lin result:",acc3)