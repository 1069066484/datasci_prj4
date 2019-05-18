# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: some global definitions. This script should
            include no other scripts in the project.
"""

import os
from enum import IntEnum


def mk_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)
    return name


# top directories
PATH_RESNET50_FOLDER = '../Office-Home_resnet50'


PATH_SAVING = mk_dir('Lsaving')
PATH_MNIST = os.path.join(PATH_SAVING, 'MNIST_data')
PATH_USPS1 = os.path.join(PATH_SAVING, 'usps/USPStrainingdata.mat')
PATH_USPS2 = os.path.join(PATH_SAVING, 'usps/USPStestingdata.mat')
PATH_RAW_DL_FEATURES = mk_dir(os.path.join(PATH_SAVING, 'raw_dls'))
PATH_ADDA_SAVING = mk_dir(os.path.join(PATH_SAVING, 'adda'))
PATH_DANN_SAVING = mk_dir(os.path.join(PATH_SAVING, 'dann'))

class DA(IntEnum):
    A2R = 0
    C2R = 1
    P2R = 2
    
DA_A2R_filenames = ['Art_Art','Art_RealWorld']
DA_C2R_filenames = ['Clipart_Clipart','Clipart_RealWorld']
DA_P2R_filenames = ['Product_Product','Product_RealWorld']
DA_filenames = [DA_A2R_filenames, DA_C2R_filenames, DA_P2R_filenames]


if __name__=='__main__':
    print(DA.filenames[0])

