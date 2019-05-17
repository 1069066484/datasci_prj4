# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:23:31 2019

@author: 12709
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import Ldata_helper as data_helper
import Lglobal_defs as global_defs
from sklearn import svm
if __name__ == '__main__':
    [src,dst] = data_helper.read_paired_labeled_features(global_defs.DA.A2R)
    Xs = src[0]
    Ys = src[1]
    Xt = dst[0]
    Yt = dst[1]
    ys_unique = np.unique(Ys)
    print(ys_unique)
    print(ys_unique.shape)