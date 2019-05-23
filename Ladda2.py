# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: Implementation of ADDA network.
            The codes are greatly referring to pumppikano's work(https://github.com/pumpikano/tf-ADDA/).
"""
import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE

from Lflip_gradient import flip_gradient
from Lutils import *

from tensorflow.examples.tutorials.mnist import input_data
import Ldata_helper as data_helper
import Lglobal_defs as global_defs

import tensorflow.contrib.slim as slim


class ADDA:
    def __init__(self, src_dl, tgt_dl, opt_step=0.001, batch_size=256, keep_prob=0.5,
                 #encoder_h_layers=[1024, 500, 500], clf_h_layers=[128], da_h_layers=[500, 500],
                 encoder_h_layers=[500, 500], clf_h_layers=[64], da_h_layers=[300, 500],
                 l2=0.00001):
        self.opt_step = opt_step
        self.l2 = l2

        src_dl[1] = data_helper.labels2one_hot(src_dl[1])
        tgt_dl[1] = data_helper.labels2one_hot(tgt_dl[1])
        
        self.tgt_tr_dl, self.tgt_te_dl = data_helper.labeled_data_split(tgt_dl, 0.6)
        self.src_tr_dl, self.src_te_dl = data_helper.labeled_data_split(src_dl, 0.6)

        self.input_dim_tgt = self.tgt_tr_dl[0].shape[1]
        self.input_dim_src = self.src_tr_dl[0].shape[1]
        self.label_count = self.src_tr_dl[1].shape[1]

        self.batch_size = batch_size

        self.encoder_h_layers = encoder_h_layers
        self.clf_h_layers = clf_h_layers
        self.da_h_layers = da_h_layers

        self.dsc_curr_batch_idx = 0
        self.clf_curr_batch_idx = 0

        #num_test = 500
        #self.comb_te_imgs = np.vstack([src_dl[0][:num_test], tgt_dl[0][:num_test]])
        #self.comb_te_labels = np.vstack([src_dl[1][:num_test], tgt_dl[1][:num_test]])
        #self.comb_te_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
        #        np.tile([0., 1.], [num_test, 1])])

        self.kp = keep_prob
        self._build_model()

    def _add_layer(self, input, out_dim, act_fun=tf.nn.relu):
        input_dim = input.shape.as_list()[1]
        w = weight_variable([input_dim, out_dim])
        b = bias_variable([out_dim])
        raw_output = tf.matmul(input, w) + b
        return raw_output if act_fun is None else act_fun(raw_output)

    def fc(self, input, layer, scope, layer_idx):
        return slim.fully_connected(input,layer,scope=scope + str(layer_idx),
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    biases_initializer=tf.random_uniform_initializer(-0.3,0.3),
                                    weights_regularizer=slim.l2_regularizer(self.l2),
                                    activation_fn=tf.nn.sigmoid
                                    )

    def _build_model(self):
        self.src_x = tf.placeholder(tf.float32, [None, self.input_dim_src], name='src_x')
        self.tgt_x = tf.placeholder(tf.float32, [None, self.input_dim_tgt], name='tgt_x')
        self.clf_y = tf.placeholder(tf.float32, [None, self.label_count], name='clf_y')

        self.src_is_tr = tf.placeholder(tf.bool, [], name='src_is_tr')
        self.tgt_is_tr = tf.placeholder(tf.bool, [], name='tgt_is_tr')
        self.clf_is_tr = tf.placeholder(tf.bool, [], name='clf_is_tr')
        self.clf_use_src = tf.placeholder(tf.bool, [], name='clf_use_src')
        self.dsc_is_tr = tf.placeholder(tf.bool, [], name='dsc_is_tr')

        batch_size = self.batch_size
        
        net = self.src_x
        for idx, layer in enumerate(self.encoder_h_layers):
            #net=slim.fully_connected(net,layer,scope="src_ec" + str(idx))
            net = self.fc(net, layer, 'src_ec', idx)
            net=slim.dropout(net,is_training=self.src_is_tr,scope="src_ec_drop" + str(idx), keep_prob=self.kp)
        src_ec_yo = net


        net = self.tgt_x
        for idx, layer in enumerate(self.encoder_h_layers):
            #net=slim.fully_connected(net,layer,scope="tgt_ec" + str(idx))
            net = self.fc(net, layer, 'tgt_ec', idx)
            net=slim.dropout(net,is_training=self.tgt_is_tr,scope="tgt_ec_drop" + str(idx), keep_prob=self.kp)
        tgt_ec_yo = net


        net = tf.cond(self.clf_use_src, lambda: src_ec_yo, lambda: tgt_ec_yo)
        for idx, layer in  enumerate(self.clf_h_layers):
            #net=slim.fully_connected(net,layer,scope="clf" + str(idx))
            net = self.fc(net, layer, 'clf', idx)
            net=slim.dropout(net,is_training=self.clf_is_tr,scope="clf_drop" + str(idx), keep_prob=self.kp)
        #clf_yo = slim.fully_connected(net,self.label_count,scope="clf_out")
        clf_yo = self.fc(net, self.label_count, 'clf_out', 0)
        la = tf.losses.get_regularization_losses()
        self.clf_loss = tf.add_n( tf.losses.get_regularization_losses() +
                                   [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                       labels=self.clf_y, logits=clf_yo))])
        clf_pred = tf.equal(tf.argmax(self.clf_y,axis=1),tf.argmax(clf_yo,axis=1))
        self.clf_acc = tf.reduce_mean(tf.cast(clf_pred,tf.float32))
           

        net = tf.concat([src_ec_yo, tgt_ec_yo], 0)
        #labels = tf.constant([1 for _ in range(int(batch_size))] +
        #                    [0 for _ in range(int(batch_size))], dtype=tf.int32)
        labels = tf.concat( [tf.zeros([tf.shape(src_ec_yo)[0]], tf.int64),  
                             tf.ones([tf.shape(tgt_ec_yo)[0]], tf.int64)], 0)
        for idx, layer in enumerate(self.da_h_layers):
            #net=slim.fully_connected(net,layer,scope="dsc" + str(idx))
            net = self.fc(net, layer, 'dsc', idx)
            net=slim.dropout(net,is_training=self.dsc_is_tr,scope="dsc_drop" + str(idx), keep_prob=self.kp)
        #dcs_yo = slim.fully_connected(net,2,scope="dsc_out")
        dcs_yo = self.fc(net, 2, 'dsc_out', 0)

        self.dsc_loss = tf.add_n(   tf.losses.get_regularization_losses() +
                                   [tf.losses.sparse_softmax_cross_entropy(
                                       labels=labels, logits=dcs_yo)])
        dsc_pred = tf.equal(labels,tf.argmax(dcs_yo,axis=1)) # FAKE - 1
        self.dsc_acc =  tf.reduce_mean(tf.cast(dsc_pred,tf.float32))


        self.tgt_ec_loss = tf.add_n( tf.losses.get_regularization_losses() + [  
                                   tf.losses.sparse_softmax_cross_entropy(
                                       labels=1-labels, logits=dcs_yo)])


    def _next_batch_dsc(self):
        batch_sz = self.batch_size
        indices = list(range(self.dsc_curr_batch_idx, self.dsc_curr_batch_idx+batch_sz))
        self.dsc_curr_batch_idx = (batch_sz + self.dsc_curr_batch_idx) % \
                        (self.src_tr_dl[0].shape[0] * self.tgt_tr_dl[0].shape[0])
        indices_src = [i%self.src_tr_dl[0].shape[0] for i in indices]
        indices_tgt = [i%self.tgt_tr_dl[0].shape[0] for i in indices]
        return [[self.src_tr_dl[0][indices_src], self.src_tr_dl[1][indices_src]],
                [self.tgt_tr_dl[0][indices_tgt], self.tgt_tr_dl[1][indices_tgt]]]


    def _next_batch_clf(self):
        batch_sz = self.batch_size
        indices = list(range(self.clf_curr_batch_idx, self.clf_curr_batch_idx+batch_sz))
        self.clf_curr_batch_idx = (batch_sz + self.clf_curr_batch_idx) % self.src_tr_dl[0].shape[0]
        indices = [i-self.src_tr_dl[0].shape[0] if i >= self.src_tr_dl[0].shape[0] else i for i in indices]
        return [self.src_tr_dl[0][indices], self.src_tr_dl[1][indices]]

        
    def train(self, iterations=1000):
        optimizer_clf = tf.train.AdamOptimizer(self.opt_step,beta1=0.5,beta2=0.999).minimize(self.clf_loss)
        optimizer_dsc = tf.train.AdamOptimizer(self.opt_step*10,beta1=0.5,beta2=0.999).minimize(self.dsc_loss)
        optimizer_tgt = tf.train.AdamOptimizer(self.opt_step*10,beta1=0.5,beta2=0.999).minimize(self.tgt_ec_loss)
        with tf.Session() as self._sess:
            self._sess.run(tf.global_variables_initializer())
            
            clf_hist = []
            for i in range(iterations):
                clf_x, clf_y = self._next_batch_clf()
                clf_dict = {self.src_x: clf_x, self.clf_y: clf_y, self.src_is_tr: True, 
                            self.tgt_x: np.zeros([1,int(self.tgt_x.shape[1])]),
                            self.tgt_is_tr: False, self.clf_is_tr: True, self.clf_use_src: True,
                            self.dsc_is_tr: False}
                optimizer_clf.run(feed_dict=clf_dict)
                if i % 500 == 0 or i == iterations-1:
                    clf_dict.update({self.clf_is_tr: False, self.src_is_tr: False})
                    train_acc = self.clf_acc.eval(feed_dict=clf_dict)
                    loss = self.clf_loss.eval(feed_dict=clf_dict)
                    clf_dict.update({self.src_x: self.src_te_dl[0], self.clf_y: self.src_te_dl[1], 
                                     self.src_is_tr:False, self.clf_is_tr: False})
                    test_acc = self.clf_acc.eval(feed_dict=clf_dict)
                    print("Classifier",i, loss, train_acc, test_acc)
                    clf_hist.append([loss, train_acc, test_acc])

            gan_hist = []
            iterations *= 100
            for i in range(iterations):
                src_dl, tgt_dl = self._next_batch_dsc()
                dsc_dict = {self.src_x: src_dl[0], self.clf_y: np.zeros([1,int(self.clf_y.shape[1])]), 
                            self.src_is_tr: False, self.tgt_x: tgt_dl[0], self.tgt_is_tr: False, 
                            self.clf_is_tr: False, self.clf_use_src: False, self.dsc_is_tr: True}
                optimizer_dsc.run(feed_dict=dsc_dict)
                tgt_dict = {self.src_x: src_dl[0], self.clf_y: np.zeros([1,int(self.clf_y.shape[1])]), 
                            self.src_is_tr: False, self.tgt_x: tgt_dl[0], self.tgt_is_tr: True, 
                            self.clf_is_tr: False, self.clf_use_src: False, self.dsc_is_tr: False}
                optimizer_tgt.run(feed_dict=tgt_dict)
                if i % 500 == 0 or i == iterations-1:
                    dsc_dict.update({self.dsc_is_tr: False})
                    dsc_loss = self.dsc_loss.eval(feed_dict=dsc_dict)
                    tgt_loss = self.tgt_ec_loss.eval(feed_dict=tgt_dict)
                    dsc_tr_acc = self.dsc_acc.eval(feed_dict=dsc_dict)
                    dsc_dict.update({self.src_x: self.src_tr_dl[0], self.tgt_x: self.tgt_tr_dl[0]})
                    dsc_te_acc = self.dsc_acc.eval(feed_dict=dsc_dict)

                    clf_dict.update({self.tgt_x: self.tgt_tr_dl[0], self.clf_y:self.tgt_tr_dl[1], 
                                     self.clf_use_src:False})
                    tgt_clf_tr_acc = self.clf_acc.eval(feed_dict=clf_dict)
                    clf_dict.update({self.tgt_x: self.tgt_te_dl[0], self.clf_y:self.tgt_te_dl[1], 
                                     self.clf_use_src:False})
                    tgt_clf_te_acc = self.clf_acc.eval(feed_dict=clf_dict)

                    '''
                    clf_dict.update({self.src_x: self.tgt_tr_dl[0], self.clf_y:self.tgt_tr_dl[1], 
                                     self.clf_use_src:True})
                    src_clf_tr_acc = self.clf_acc.eval(feed_dict=clf_dict)
                    clf_dict.update({self.src_x: self.tgt_te_dl[0], self.clf_y:self.tgt_te_dl[1], 
                                     self.clf_use_src:True})
                    src_clf_te_acc = self.clf_acc.eval(feed_dict=clf_dict)
                    '''
                    src_clf_tr_acc = -1
                    src_clf_te_acc = -1
                    print('GAN', dsc_loss, tgt_loss, dsc_tr_acc, dsc_te_acc, tgt_clf_tr_acc, 
                          tgt_clf_te_acc, src_clf_tr_acc, src_clf_te_acc)
                    gan_hist.append([dsc_loss, tgt_loss, dsc_tr_acc, dsc_te_acc, tgt_clf_tr_acc, 
                          tgt_clf_te_acc, src_clf_tr_acc, src_clf_te_acc])


            """
            self.src_x = tf.placeholder(tf.float32, [None, self.input_dim])
            self.tgt_x = tf.placeholder(tf.float32, [None, self.input_dim])
            self.clf_y = tf.placeholder(tf.float32, [None, self.label_count])

            self.src_is_tr = tf.placeholder(tf.bool, [])
            self.tgt_is_tr = tf.placeholder(tf.bool, [])
            self.clf_is_tr = tf.placeholder(tf.bool, [])
            self.clf_use_src = tf.placeholder(tf.bool, [])
            self.dsc_is_tr = tf.placeholder(tf.bool, [])
            """

def main():
    [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(global_defs.DA.A2R)
    adda = ADDA(src_dl, tgt_dl)
    adda.train(iterations=1000)


def main2():
    mnist_dl = data_helper.read_mnist_dl()
    usps_dl = data_helper.read_usps_dl()
    adda = ADDA(mnist_dl, usps_dl)
    adda.train(iterations=1000)


if __name__=='__main__':
    main2()

"""

# 定义占位符，X表示网络的输入，Y表示真实值label
X = tf.placeholder("float", [None, 224, 224, 3])
Y = tf.placeholder("float", [None, 100])
 
#调用含batch_norm的resnet网络，其中记得is_training=True
logits = model.resnet(X, 100, is_training=True)
cross_entropy = -tf.reduce_sum(Y*tf.log(logits))
 
#训练的op一定要用slim的slim.learning.create_train_op，只用tf.train.MomentumOptimizer.minimize（）是不行的
opt = tf.train.MomentumOptimizer(lr_rate, 0.9)
train_op = slim.learning.create_train_op(cross_entropy, opt, global_step=global_step)
 
#更新操作，具体含义不是很明白，直接套用即可
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    updates = tf.group(*update_ops)
    cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

"""