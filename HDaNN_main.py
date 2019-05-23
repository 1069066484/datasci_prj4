# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:49:33 2019

@author: 12709
"""

import HDaNN
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm
import Lglobal_defs as global_defs
import Ldata_helper as data_helper
from sklearn import preprocessing

#import data_loader
import Hmmd

global scaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#batch_size = 1
LEARNING_RATE = 0.05
MOMEMTUN = 0.05
L2_WEIGHT = 0.003
DROPOUT = 0.5
N_EPOCH = 30
BATCH_SIZE = [64, 64]
LAMBDA = 1
GAMMA = 10 ^ 3
RESULT_TRAIN = []
RESULT_TEST = []
log_train = open('train_l1_bs64_dr0.5_lin.txt', 'w')
log_test = open('test_l1_bs64_dr0.5_lin.txt', 'w')


class MyTrainData_src(Dataset):
    def __init__(self, i):
        [src, tar] = data_helper.read_paired_labeled_features(i)
        self.src_feature = src[0]
        self.src_label = src[1]
        global scaler
        scaler = preprocessing.StandardScaler().fit(self.src_feature)
        self.src_feature = scaler.transform(self.src_feature)
        
    def __len__(self):
        return len(self.src_label)
    
    def __getitem__(self, index):
        src_data = self.src_feature[index]
        src_label = self.src_label[index]
        src_label = torch.tensor(src_label).long()
        src_data  = torch.tensor(src_data)
        src_data = src_data.float()
        return src_data, src_label
    
class MyTrainData_tar(Dataset):
    def __init__(self, i):
        [src, tar] = data_helper.read_paired_labeled_features(i)
        self.tar_feature = tar[0]
        self.tar_label = tar[1]
        scale = preprocessing.StandardScaler().fit(self.tar_feature)
        #global scaler
        self.tar_feature = scale.transform(self.tar_feature)
        
    def __len__(self):
        return len(self.tar_label)
    
    def __getitem__(self, index):
        tar_data = self.tar_feature[index]
        tar_label = self.tar_label[index]
        tar_label = torch.tensor(tar_label).long()
        tar_data = torch.tensor(tar_data)
        tar_data = tar_data.float()
        return tar_data, tar_label
            
def mmd_loss(x_src, x_tar):
    #print(x_src.size(0))
    #print(x_tar.size(0))
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])


def train(model, optimizer, epoch, data_src, data_tar):
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    batch_j = 0
    list_tar = list(enumerate(data_tar))
    for batch_id, (data, target) in enumerate(data_src):
        _, (x_tar, y_target) = list_tar[batch_j]
        data, target = data.data.view(-1, 2048).to(DEVICE), target.to(DEVICE)
        #data = data.float()
        x_tar, y_target = x_tar.view(-1, 2048).to(DEVICE), y_target.to(DEVICE)
        #x_tar = x_tar.float()
        model.train()
        y_src, x_src_mmd, x_tar_mmd = model(data, x_tar)

        loss_c = criterion(y_src, target)
        loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
        pred = y_src.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss = loss_c + LAMBDA * loss_mmd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_train += loss.data
        if batch_j<len(list_tar)-1:
            batch_j +=1
        res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
            epoch, N_EPOCH, batch_id + 1, len(data_src), loss.data
        )
    total_loss_train /= len(data_src)
    acc = float(correct) / len(data_src.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}'.format(
        epoch, N_EPOCH, total_loss_train, correct, len(data_src.dataset), acc
    )
    res1 = '{:} {:.6f} {:.4f}'.format(epoch,total_loss_train,acc)
    tqdm.write(res_e)
    log_train.write(res1 + '\n')
    RESULT_TRAIN.append([epoch, total_loss_train, acc])
    return model


def test(model, data_tar, e):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_tar):
            data, target = data.view(-1,2048).to(DEVICE),target.to(DEVICE)
            model.eval()
            ypred, _, _ = model(data, data)
            loss = criterion(ypred, target)
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total_loss_test += loss.data
        accuracy = float(correct) / len(data_tar.dataset)
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}'.format(
            total_loss_test, correct, len(data_tar.dataset), accuracy
        )
        res1 = '{:} {:.6f} {:.4f}'.format(e,total_loss_test,accuracy)
    tqdm.write(res)
    RESULT_TEST.append([e, total_loss_test, accuracy])
    log_test.write(res1 + '\n')


if __name__ == '__main__':
    #rootdir = '../../../data/office_caltech_10/'
    torch.manual_seed(1)
    i = 2
    data_src = DataLoader(dataset = MyTrainData_src(i),batch_size=BATCH_SIZE[0],shuffle=True, drop_last= True)
    data_tar = DataLoader(dataset = MyTrainData_tar(i),batch_size=BATCH_SIZE[1],shuffle=True, drop_last= True)
    '''
    data_src = data_loader.load_data(
        root_dir=rootdir, domain='amazon', batch_size=BATCH_SIZE[0])
    data_tar = data_loader.load_test(
        root_dir=rootdir, domain='webcam', batch_size=BATCH_SIZE[1])
    '''
    model = DaNN.DaNN(n_input=2048, n_hidden=256, n_class=65)
    model = model.to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMEMTUN,
        weight_decay=L2_WEIGHT
    )
    for e in tqdm(range(1, N_EPOCH + 1)):
        model = train(model=model, optimizer=optimizer,
                      epoch=e, data_src=data_src, data_tar=data_tar)
        test(model, data_tar, e)
    torch.save(model, 'model_dann.pkl')
    log_train.close()
    log_test.close()
    res_train = np.asarray(RESULT_TRAIN)
    res_test = np.asarray(RESULT_TEST)
    np.savetxt('res_train_a-w.csv', res_train, fmt='%.6f', delimiter=',')
    np.savetxt('res_test_a-w.csv', res_test, fmt='%.6f', delimiter=',')