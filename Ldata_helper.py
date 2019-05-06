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


def npfilename(filename):
    return posfix_filename(filename, '.npy')


def pkfilename(filename):
    return posfix_filename(filename, '.pkl')


def posfix_filename(filename, postfix):
    if not filename.endswith(postfix):
        filename += postfix
    return filename


def read_features(use_dl=False):
    """
    37322 is the number of sample images. 2048 is number of deep features
    :return: a 37322-by-2048 ndarray
    DEPRECATED! call read_sift_labeled_features instead.
    """
    if use_dl:
        path_np_features = os.path.join(global_defs.PATH_FEATURES_FOLDER, 'features32')
        path_np_features = npfilename(path_np_features)
        if os.path.exists(path_np_features):
            return np.load(path_np_features)
        feature_arr = []
        with open(global_defs.PATH_FEATURES, 'r') as file:
            for line in file:
                line = line.strip()
                feature_arr.append(line.split(' '))
        feature_arr = np.array(feature_arr, dtype=np.float32)
        np.save(path_np_features, feature_arr)
        return feature_arr
    else:
        #import BOW
        #return BOW.read_bowed_sift_features()
        raise Exception("Trying to read SIFT features via read_features() call")


def read_labels():
    if hasattr(read_labels,'labels'):
        return read_labels.labels
    labels = []
    for line in open(global_defs.PATH_LABELS, 'r'):
        labels.append(int(line))
    labels = np.array(labels)
    read_labels.labels = labels
    return labels


def read_bowed_labeled_features(use_dl=False, use_cv=False):
    """
    use_dl: use deep learning features or BOW features
    :return a list [features, labels]
    """
    if use_dl:
        features = read_features(use_dl = True)
        labels = []
        for line in open(global_defs.PATH_LABELS, 'r'):
            labels.append(int(line))
        labels = np.array(labels)
    else:
        if not use_cv:
            import BOW
            return BOW.read_BOWed_labeled_features()
        else:
            import sift_bow_cv
            return sift_bow_cv.read_labeled_BOWed_features_cv()
    return [features, labels]


def read_sift_labeled_features(use_dl=False):
    """
    DEPRECATED! This function read truncated sift features! Invoke read_part_sift_features_images if you want to
        train a classifier. Invoke sift_feature_traverse if you want to access sift feature of each images.
    use_dl: use deep learning features or sift features
    ATTENTION:  with use_dl set false, sift features would be read, sift features is a N-by-m-by-128 matrix.
                where N is the number of samples(not neccessarily the same as samples of deep features), m is the
                number of selected descriptors. labels is a N-element array.
                e.g.
                import data_helper
                [features,labels] = data_helper.read_sift_labeled_features()
                print(features.shape)   # (N,m,128)
                print(labels.shape)     # (N,)
    :return a list [features, labels]
    """
    if use_dl:
        features = read_features(use_dl = True)
        labels = []
        for line in open(global_defs.PATH_LABELS, 'r'):
            labels.append(int(line))
        labels = np.array(labels)
    else:
        import sift_feature
        [features, labels] = sift_feature.read_sift_features()
    return [features, labels]


_dicts = None
def get_dicts():
    """
    return two dicts [id2name, name2id] dicts
    """
    global _dicts
    if _dicts is not None:
        return _dicts
    if os.path.exists(global_defs.PATH_SAVING_DICTS):
        _dicts = pickle.load(open(global_defs.PATH_SAVING_DICTS, 'rb+'))
        return _dicts
    # print("Create")
    file = open(global_defs.PATH_CLASSES, 'r')
    id2name = {}
    name2id = {}
    for line in file:
        line = line.strip()
        line = line.split('\t')
        id = int(line[0])
        name = line[1]
        id2name[id] = name
        name2id[name] = id
    _dicts = [id2name, name2id]
    pickle.dump(_dicts, open(global_defs.PATH_SAVING_DICTS, 'wb+'))
    return _dicts


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


def img_traverse(class_id=-1):
    if not 'img_traverse.dict' in locals():
        img_traverse.name2id = get_dicts()[1]
    for dir in os.listdir(global_defs.PATH_IMAGES_FOLDER):
        class_dir = os.path.join(global_defs.PATH_IMAGES_FOLDER, dir)
        if os.path.isdir(class_dir):
            id = img_traverse.name2id[dir]
            if class_id == -1 or class_id == id:
                for file in os.listdir(class_dir):
                    if file.endswith('.jpg') or file.endswith('.png'):
                        file = os.path.join(class_dir, file)
                        yield [file, id, class_dir]


def sift_feature_traverse(mode='per_class'):
    """
    sift_feature_traverse can be used for 'in' loop to acquire sift features.
    @Param-mode: can be one of per_class(default), per_image and per_descriptor.
        mode of 'per_class' returns 
            [pickle_file_name_of_sift_feature_of_each_class,
            class_id,
            sift_feature_of_each_class]
                sift_feature_of_each_class is a list of n matrix, each matrix, 
            m-by-k, contains descriptors of an image of the class, where m the 
            number of descriptors of the image and k=128. m can be different 
            for different images.
        mode of 'per_image' returns
            [class_id,
            descriptors_of_each_image]
                descriptors_of_each_image is a m-by-k matrix containing 
            descriptors of an image of the class indicated by class_id.
        mode of 'per_descriptor' returns
            each descriptor of all images. each descriptor is a k-element array,
            where k=128.
    For example:
        Codes:
            for file, id, sift_feature_of_class in sift_feature_traverse('per_class'):
                print(file, id, len(sift_feature_of_class))
            for id, sift_feature_per_image in sift_feature_traverse('per_image'):
                print(id, sift_feature_per_image.shape)
            for sift_feature_per_descriptor in sift_feature_traverse('per_descriptor'):
                print(sift_feature_per_descriptor.shape)
        Output:
            Animals_with_Attributes2/JPEGImages\antelope\1.pkl 1 1046
            Animals_with_Attributes2/JPEGImages\bat\30.pkl 30 383
            Animals_with_Attributes2/JPEGImages\beaver\4.pkl 4 193
            ... ...
            1 (6756, 128)
            1 (4406, 128)
            1 (4391, 128)
            1 (8011, 128)
            ... ...
            (128,)
            (128,)
            (128,)
            (128,)
            (128,)
    """
    if not 'sift_feature_traverse.dict' in locals():
        sift_feature_traverse.name2id = get_dicts()[1]
    for dir in os.listdir(global_defs.PATH_IMAGES_FOLDER):
        class_dir = os.path.join(global_defs.PATH_IMAGES_FOLDER, dir)
        if os.path.isdir(class_dir):
            id = sift_feature_traverse.name2id[dir]
            sift_feature_file = os.path.join(class_dir, str(id))
            sift_feature_file = pkfilename(sift_feature_file)
            sift_feature_per_class = pickle.load(open(sift_feature_file,'rb'))
            if mode == 'per_class':
                yield [sift_feature_file, id, sift_feature_per_class]
            else: 
                for sift_feature_per_image in sift_feature_per_class:
                    if mode == 'per_image':
                        yield [id, sift_feature_per_image]
                    elif mode == 'per_descriptor':
                        for sift_feature_per_descriptor in sift_feature_per_image:
                            yield sift_feature_per_descriptor
                    else:
                        raise Exception('sift_feature_traverse error. Invalid parameter \'mode\' = ' + mode, "'mode' should be one among: 'per_class', 'per_image' and 'per_descriptor'")


def read_part_sift_features_images(rate=0.05):
    """
    read_part_sift_features_images select a certain percent, indicated by rate, of images from 
        each class and extract their sift_feature matrix.
    For example,
        Codes:
            part_sift_feature_images = read_part_sift_features_images()
            print(len(part_sift_feature_images))
        Outputs:
            37320
    """
    import sift_feature
    return sift_feature.read_part_sift_features2(rate)[1]


def read_part_sift_descriptors(rate=0.05):
    """
    read_part_sift_descriptors select descriptors from a certain percent, indicated by rate, of
        sift features of images from each class. Actually, the descriptors are corresponding to 
        that of read_part_sift_features_images(rate)
    For example,
        Codes:
            part_sift_descriptors = read_part_sift_descriptors()
            print(part_sift_descriptors.shape)
        Outputs:
            (3236336, 128)
    """
    import BOW
    return BOW.read_list_sift_features(rate)


if __name__ == '__main__':
    part_sift_feature_images = read_part_sift_features_images()
    print(len(part_sift_feature_images))
    part_sift_descriptors = read_part_sift_descriptors()
    print(part_sift_descriptors.shape)