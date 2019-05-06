# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: some global definitions. This script should
            include no other scripts in the project.
"""

import os


NUM_TOTAL_ANIMALS = 50
NUM_TOTAL_FEATURES = 85


# top directories
PATH_BASE_FOLDER = 'Animals_with_Attributes2'
PATH_FEATURES_FOLDER = 'Features/ResNet101'
PATH_IMAGES_FOLDER = 'Animals_with_Attributes2/JPEGImages'


# 50 rows: [class_number class_name]
PATH_CLASSES = os.path.join(PATH_BASE_FOLDER, 'classes.txt')
# 50 rows: [0 1 0 1 0 1 0 0 0 0 1 ...]
PATH_PREDICATE_MATRIX_BINARY = os.path.join(PATH_BASE_FOLDER, 'predicate-matrix-binary.txt')
# 50 rows: [10.2 0.33 0.2 60.1 -1.2 ...]
PATH_PREDICATE_MATRIX_CONTINUOUS = os.path.join(PATH_BASE_FOLDER, 'predicate-matrix-continuous.txt')
# 85 rows: [feature_number feature_name]
PATH_PREDICATES = os.path.join(PATH_BASE_FOLDER, 'predicates.txt')


# 700+M, float numbers
PATH_FEATURES = os.path.join(PATH_FEATURES_FOLDER, 'AwA2-features.txt')
# many rows, [horse_11222.jpg]
PATH_FILENAMES = os.path.join(PATH_FEATURES_FOLDER, 'AwA2-filenames.txt')
# many rows, (same as FILENAMES), [1/0]
PATH_LABELS = os.path.join(PATH_FEATURES_FOLDER, 'AwA2-labels.txt')


PATH_RESULTS_FOLDER = 'results'
if not os.path.exists(PATH_RESULTS_FOLDER):
    os.mkdir(PATH_RESULTS_FOLDER)
PATH_RESULT_BOWED_FEATURES = os.path.join(PATH_RESULTS_FOLDER, 'bowed_feature')
PATH_RESULT_BOWED_CV_FEATURES = os.path.join(PATH_RESULTS_FOLDER, 'bowed_feature_cv')
PATH_RESULT_SIFT_FEATURE = os.path.join(PATH_RESULTS_FOLDER,'sifted')
PATH_RESULT_SIFT_PCA_FEATURE = os.path.join(PATH_RESULTS_FOLDER,'sifted_pca')
PATH_RESULT_SIFT_LABELS = os.path.join(PATH_RESULTS_FOLDER, 'sift_labels')


PATH_SAVING_FOLDER = 'saving'
if not os.path.exists(PATH_SAVING_FOLDER):
    os.mkdir(PATH_SAVING_FOLDER)
PATH_SAVING_DICTS = os.path.join(PATH_SAVING_FOLDER, 'dicts')
PATH_SAVING_MINIKMEANS_CLUSTERER = os.path.join(PATH_SAVING_FOLDER, 'mini_kmeans')
PATH_SAVING_LIST_SIFT = os.path.join(PATH_SAVING_FOLDER, 'list_sifted')
PATH_SAVING_BOW_DICT_CV = os.path.join(PATH_SAVING_FOLDER, 'bow_dict_cv')


if __name__=='__main__':
    print(os.path.exists(PATH_BASE_FOLDER))

