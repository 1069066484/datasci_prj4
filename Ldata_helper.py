# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: some help functions.
"""

import Lglobal_defs as global_defs
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import csv


def read_mnist(one_hot=True):
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(os.path.join(global_defs.PATH_MNIST),one_hot=one_hot)


def posfix_filename(filename, postfix):
    if not filename.endswith(postfix):
        filename += postfix
    return filename


def npfilename(filename):
    return posfix_filename(filename, '.npy')


def pkfilename(filename):
    return posfix_filename(filename, '.pkl')


def csvfilename(filename):
    return posfix_filename(filename, '.csv')


def csvfile2nparr(csvfn):
    csvfn = csvfilename(csvfn)
    csvfn = csv.reader(open(csvfn,'r'))
    def read_line(line):
        return [float(i) for i in line]
    m = [read_line(line) for line in csvfn]
    return np.array(m)


def read_labeled_features(csvfn):
    arr = csvfile2nparr(csvfn)
    data, labels = np.hsplit(arr,[-1])
    labels = labels.reshape(labels.size)
    return [data, labels]


def read_paired_labeled_features(type_DA):
    src_csv, dst_csv = global_defs.DA_filenames[type_DA]
    path_src_dst_da_lf = pkfilename(src_csv + "__" + dst_csv)
    path_src_dst_da_lf = os.path.join(global_defs.PATH_RAW_DL_FEATURES, path_src_dst_da_lf)
    if os.path.exists(path_src_dst_da_lf):
        return list(pickle.load(open(path_src_dst_da_lf, 'rb')))
    src_csv = csvfilename(os.path.join(global_defs.PATH_RESNET50_FOLDER, src_csv))
    dst_csv = csvfilename(os.path.join(global_defs.PATH_RESNET50_FOLDER, dst_csv))
    src_dst_da_lf = [list(read_labeled_features(src_csv)), list(read_labeled_features(dst_csv))]
    pickle.dump(src_dst_da_lf, open(path_src_dst_da_lf, 'wb'))
    return list(src_dst_da_lf)


def plt_show_it_data(it_data, xlabel='iterations', ylabel=None, title=None, do_plt_last=True):
    y = it_data
    x = list(range(len(y)))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel('' if ylabel is None else ylabel)
    plt.title('' if title is None else title)
    if do_plt_last:
        plt.text(x[-1], y[-1], y[-1])
    plt.show()


def plt_show_scatter(xs, ys, xlabel=None, ylabel=None, title=None):
    colors = ['r', 'y', 'k', 'g', 'b', 'm']
    num2plt = min(len(colors), len(xs))
    for i in range(num2plt):
        plt.scatter(x=xs[i], y=ys[i], c=colors[i], marker='.')
    plt.xlabel('' if xlabel is None else xlabel)
    plt.ylabel('' if ylabel is None else ylabel)
    plt.title('' if title is None else title)
    plt.show()


def non_repeated_random_nums(nums, num):
    num = math.ceil(num)
    nums = np.random.permutation(nums)
    return nums[:num]


def index_split(num, percent1):
    percent1 = math.ceil(num * percent1)
    nums = np.random.permutation(num)
    return [nums[:percent1], nums[percent1:]]


def labeled_data_split(labeled_data, percent_train=0.6):
    train_idx, test_idx = index_split(labeled_data[0].shape[0], percent_train)
    train_ld = [labeled_data[0][train_idx], labeled_data[1][train_idx]]
    test_ld = [labeled_data[0][test_idx], labeled_data[1][test_idx]]
    return [train_ld, test_ld]


def rand_arr_selection(arr, num):
    nonrep_rand_nums = non_repeated_random_nums(arr.shape[0], num)
    return [arr[nonrep_rand_nums], nonrep_rand_nums]


def labels2one_hot(labels):
    labels = np.array(labels, dtype=np.int)
    if len(labels.shape) == 1:
        minl = np.min(labels)
        labels -= minl
        maxl = np.max(labels) + 1
        r = range(maxl)
        return np.array([[1 if i==j else 0 for i in r] for j in labels])
    return labels


def shuffle_labeled_data(dl):
    data, labels = dl
    a = np.arange(labels.shape[0])
    np.random.shuffle(a)
    return [data[a], labels[a]]


def _get_dicts_test():
    id2name, name2id = get_dicts()
    print(id2name, name2id)


def _test_read_paired_labeled_features():
    [src_dl, dst_dl] = read_paired_labeled_features(global_defs.DA.A2R)
    print(src_dl[0].shape)
    print(src_dl[1].shape)
    print(dst_dl[0].shape)
    print(dst_dl[1].shape)


def _test_labels_one_hot():
    a = np.array([2,1,0,0,0,2,1,1,1])
    print(labels2one_hot(a))


if __name__ == '__main__':
    _test_read_paired_labeled_features()
