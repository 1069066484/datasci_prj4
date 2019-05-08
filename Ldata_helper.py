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
    return input_data.read_data_sets(global_defs.PATH_MNIST,one_hot=one_hot)


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
    return data, labels


def read_paired_labeled_features(type_DA):
    src_csv, dst_csv = global_defs.DA_filenames[type_DA]
    path_src_dst_da_lf = pkfilename(src_csv + "__" + dst_csv)
    path_src_dst_da_lf = os.path.join(global_defs.PATH_RAW_DL_FEATURES, path_src_dst_da_lf)
    if os.path.exists(path_src_dst_da_lf):
        return pickle.load(open(path_src_dst_da_lf, 'rb'))
    src_csv = csvfilename(os.path.join(global_defs.PATH_RESNET50_FOLDER, src_csv))
    dst_csv = csvfilename(os.path.join(global_defs.PATH_RESNET50_FOLDER, dst_csv))
    src_dst_da_lf = [read_labeled_features(src_csv), read_labeled_features(dst_csv)]
    pickle.dump(src_dst_da_lf, open(path_src_dst_da_lf, 'wb'))
    return src_dst_da_lf


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

def rand_arr_selection(arr, num):
    nonrep_rand_nums = non_repeated_random_nums(arr.shape[0], num)
    return [arr[nonrep_rand_nums], nonrep_rand_nums]


def _get_dicts_test():
    id2name, name2id = get_dicts()
    print(id2name, name2id)


if __name__ == '__main__':
    [src_dl, dst_dl] = read_paired_labeled_features(global_defs.DA.A2R)
    print(src_dl[0].shape)
    print(src_dl[1].shape)
    print(dst_dl[0].shape)
    print(dst_dl[1].shape)
