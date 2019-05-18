# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: Implementation of dann network.
            Great thanks to pumppikano's work(https://github.com/pumpikano/tf-dann/).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE

from Lflip_gradient import flip_gradient
from Lutils import *

from tensorflow.examples.tutorials.mnist import input_data
import Ldata_helper as data_helper
import Lglobal_defs as global_defs
import os


class DANN:
    def __init__(self, src_dl, tgt_dl, opt_step=0.00002, batch_size=256, 
                 encoder_h_layers=[1024], clf_h_layers=[256], da_h_layers=[512, 512]
                 #encoder_h_layers=[10, 10], clf_h_layers=[10], da_h_layers=[10, 10]
                 ):
        src_dl[1] = data_helper.labels2one_hot(src_dl[1])
        tgt_dl[1] = data_helper.labels2one_hot(tgt_dl[1])
        self.opt_step = opt_step
        self.tgt_tr_dl, self.tgt_te_dl = data_helper.labeled_data_split(tgt_dl, 0.75)
        self.src_tr_dl, self.src_te_dl = data_helper.labeled_data_split(src_dl, 0.75)

        self.input_dim = self.src_tr_dl[0].shape[1]
        self.label_count = self.src_tr_dl[1].shape[1]

        self.batch_size = batch_size

        self.encoder_h_layers = encoder_h_layers
        self.clf_h_layers = clf_h_layers
        self.da_h_layers = da_h_layers

        num_test = 500
        self.comb_tr_imgs = np.vstack([self.src_tr_dl[0][:num_test], self.tgt_tr_dl[0][:num_test]])
        self.comb_tr_labels = np.vstack([self.src_tr_dl[1][:num_test], self.tgt_tr_dl[1][:num_test]])

        self.comb_te_imgs = np.vstack([self.src_te_dl[0][:num_test], self.tgt_te_dl[0][:num_test]])
        self.comb_te_labels = np.vstack([self.src_te_dl[1][:num_test], self.tgt_te_dl[1][:num_test]])
        self.comb_te_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                np.tile([0., 1.], [num_test, 1])])
        self.num_test = num_test
        self._build_model()

    def _add_layer(self, input, out_dim, act_fun=tf.nn.sigmoid):
        input_dim = input.shape.as_list()[1]
        w = weight_variable([input_dim, out_dim])
        b = bias_variable([out_dim])
        raw_output = tf.matmul(input, w) + b
        output = raw_output if act_fun is None else act_fun(raw_output)
        return output

    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.y = tf.placeholder(tf.float32, [None, self.label_count])
        self.domain = tf.placeholder(tf.float32, [None, 2])

        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])
        batch_size = self.batch_size
        X_input = self.X
        # X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.
        
        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            input = X_input
            for layer in self.encoder_h_layers:
                input = self._add_layer(input, layer)
            # The domain-invariant feature
            self.feature = input
            
        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size // 2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)
            
            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)
            
            print("tf.shape(all_labels)=",tf.shape(all_labels()))
            print("tf.shape(source_labels)=",tf.shape(source_labels()))
            #tf.shape(all_labels)= Tensor("label_predictor/Shape:0", shape=(2,), dtype=int32)
            #tf.shape(source_labels)= Tensor("label_predictor/Shape_1:0", shape=(2,), dtype=int32)

            input = classify_feats
            for layer in self.clf_h_layers:
                input = self._add_layer(input, layer)

            #h_fc0 = self._add_layer(classify_feats, 128)
            h_fc1 = self._add_layer(input, self.label_count, None)

            logits = h_fc1
            
            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)
            #feat = self.feature
            input = feat
            for layer in self.da_h_layers:
                input = self._add_layer(input, layer)

            #d_h_fc0 = self._add_layer(feat, 32) 
            d_b_fc1 = self._add_layer(input, 2, None) 

            d_logits = d_b_fc1
            
            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)


    def build_graph(self):
        self.graph = tf.get_default_graph()
        self.inited = False
        with self.graph.as_default():
            #model = MNISTModel()
            
            self.lr = tf.placeholder(tf.float32, [])
            
            self.pred_loss = tf.reduce_mean(self.pred_loss)
            self.domain_loss = tf.reduce_mean(self.domain_loss)
            self.total_loss = self.pred_loss * 2 + self.domain_loss

            #self.regular_train_op = tf.train.MomentumOptimizer(self.lr, 0.001).minimize(self.pred_loss)
            #self.dann_train_op = tf.train.MomentumOptimizer(self.lr, 0.001).minimize(self.total_loss)
            self.regular_train_op = tf.train.AdamOptimizer(self.lr,beta1=0.5,beta2=0.9).minimize(self.pred_loss)
            self.dann_train_op = tf.train.AdamOptimizer(self.lr,beta1=0.5,beta2=0.9).minimize(self.total_loss)
            
            # Evaluation
            correct_label_pred = tf.equal(tf.argmax(self.classify_labels, 1), tf.argmax(self.pred, 1))
            self.label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
            correct_domain_pred = tf.equal(tf.argmax(self.domain, 1), tf.argmax(self.domain_pred, 1))
            self.domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

    def train_eval(self, training_mode, iterations=8000, verbose=True):
        with tf.Session(graph=self.graph,config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333) ) ) as sess:
            if not self.inited:
                tf.global_variables_initializer().run()
                self.inited = True
            batch_size = self.batch_size
            # Batch generators
            gen_source_batch = batch_generator(
                self.src_tr_dl, batch_size // 2)
            gen_target_batch = batch_generator(
                self.tgt_tr_dl, batch_size // 2)
            gen_source_only_batch = batch_generator(
                self.src_tr_dl, batch_size)
            gen_target_only_batch = batch_generator(
                self.tgt_tr_dl, batch_size)

            domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                                    np.tile([0., 1.], [batch_size // 2, 1])])

            

            batch_loss = -1
            dloss = -1
            ploss = -1
            d_acc = -1
            p_acc = -1
            ret = []
            for training_mode in ['source', 'dann']:
                # Training loop
                hist = []
                for i in range(iterations):
                
                    # Adaptation param and learning rate schedule as described in the paper
                    p = float(i) / iterations
                    l = 2. / (1. + np.exp(-10. * p)) - 1
                    lr = 0.01 / (1. + 10 * p)**0.75 * 0.004

                    # Training step
                    if training_mode == 'dann':

                        X0, y0 = next(gen_source_batch)
                        X1, y1 = next(gen_target_batch)
                        X = np.vstack([X0, X1])
                        y = np.vstack([y0, y1])

                        _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                            [self.dann_train_op, self.total_loss, self.domain_loss, 
                            self.pred_loss, self.domain_acc, self.label_acc],
                            feed_dict={self.X: X, self.y: y, self.domain: domain_labels,
                                    self.train: True, self.l: l, self.lr: lr})

                        #if verbose and (i % 500 == 0 or i == iterations-1):
                        #    print(i,'loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {}'.format(
                        #            batch_loss, d_acc, p_acc, p, l, lr))

                    elif training_mode == 'source':
                        X, y = next(gen_source_only_batch)
                        _, batch_loss = sess.run([self.regular_train_op, self.pred_loss],
                                            feed_dict={self.X: X, self.y: y, self.train: False,
                                                        self.l: l, self.lr: lr})

                    elif training_mode == 'target':
                        X, y = next(gen_target_only_batch)
                        _, batch_loss = sess.run([self.regular_train_op, self.pred_loss],
                                            feed_dict={self.X: X, self.y: y, self.train: False,
                                                        self.l: l, self.lr: lr})


                    if i % 500 == 0 or i == iterations-1:
                        # Compute final evaluation on test data
                        source_te_acc = sess.run(self.label_acc,
                                            feed_dict={self.X: self.src_te_dl[0], self.y: self.src_te_dl[1],
                                                    self.train: False})

                        target_te_acc = sess.run(self.label_acc,
                                            feed_dict={self.X: self.tgt_te_dl[0], self.y: self.tgt_te_dl[1],
                                                    self.train: False})

                         # Compute final evaluation on test data
                        source_tr_acc = sess.run(self.label_acc,
                                            feed_dict={self.X: self.src_tr_dl[0], self.y: self.src_tr_dl[1],
                                                    self.train: False})

                        target_tr_acc = sess.run(self.label_acc,
                                            feed_dict={self.X: self.tgt_tr_dl[0], self.y: self.tgt_tr_dl[1],
                                                    self.train: False})

                        test_domain_acc = sess.run(self.domain_acc,
                                            feed_dict={self.X: self.comb_te_imgs,
                                                    self.domain: self.comb_te_domain, self.l: 1.0})
                        hist.append([i, source_te_acc, target_te_acc, test_domain_acc, batch_loss, dloss, ploss, d_acc, p_acc,source_tr_acc,target_tr_acc])
                        print(i,source_te_acc, target_te_acc, test_domain_acc,source_tr_acc,target_tr_acc)

                #self.comb_te_imgs = np.vstack([src_dl[0][:num_test], tgt_dl[0][:num_test]])
                te_emb = sess.run(self.feature, feed_dict={self.X: self.comb_te_imgs})
                tr_emb = sess.run(self.feature, feed_dict={self.X: self.comb_tr_imgs})
                hist = np.array(hist)
                ret.append([hist, te_emb, tr_emb])
        return ret


def plt_hist(src_hist, da_hist, path):
    plt.figure(figsize=(25,10))
    plt.subplot(122)
    da_hist[:, 4] -= np.min(da_hist[:, 4]); da_hist[:, 4] /= np.max(da_hist[:, 4])
    da_hist[:, 5] -= np.min(da_hist[:, 5]); da_hist[:, 5] /= np.max(da_hist[:, 5])
    da_hist[:, 6] -= np.min(da_hist[:, 6]); da_hist[:, 6] /= np.max(da_hist[:, 6])
    plt.plot(da_hist[:, 0], da_hist[:, 4], label='DANN total loss')
    plt.plot(da_hist[:, 0], da_hist[:, 5], label='Discriminator loss')
    plt.plot(da_hist[:, 0], da_hist[:, 6], label='classifier loss')
    plt.plot(da_hist[:, 0], da_hist[:, 7], label='discriminator accuracy')

    plt.plot(da_hist[:, 0], da_hist[:, 1], label='source accuracy(test)')
    plt.plot(da_hist[:, 0], da_hist[:, 2], label='target accuracy(test)')
    plt.plot(da_hist[:, 0], da_hist[:, -2], label='source accuracy(train)')
    plt.plot(da_hist[:, 0], da_hist[:, -1], label='target accuracy(train)')

    plt.legend(['DANN total loss', 'Discriminator loss', 'classifier loss', 'discriminator accuracy(training)', 
                'source accuracy(test)', 'target accuracy(test)', 'source accuracy(train)', 'target accuracy(train)'])
    plt.xlabel('Iterations')
    plt.title('DANN training history of da stage')
    #plt.show()
    

    plt.subplot(121)
    plt.plot(src_hist[:, 0], src_hist[:, 1], label='source accuracy(test)')
    #plt.plot(src_hist[:, 0], src_hist[:, 2], label='target accuracy(test)')
    plt.plot(src_hist[:, 0], src_hist[:, -2], label='source accuracy(train)')
    #plt.plot(src_hist[:, 0], src_hist[:, -1], label='target accuracy(train)')
    plt.plot(src_hist[:, 0], src_hist[:, 3], label='domain accuracy(test)')
    src_hist[:, 4] -= np.min(src_hist[:, 4]); src_hist[:, 4] /= np.max(src_hist[:, 4])
    plt.plot(src_hist[:, 0], src_hist[:, 4], label='classifier loss')
    plt.legend(['source accuracy(test)', 
                #'target accuracy(test)', 
                'source accuracy(train)', 
                #'target accuracy(train)',
                'domain accuracy(test)', 'classifier loss'])
    plt.xlabel('Iterations')
    plt.title('DANN training history of source stage')
    plt.savefig(os.path.join(path, 'dann.png'))


def visualize_da2(src_data, tgt_data, title=None, figname=None):
    plt.figure(figsize=(15,10))
    src_data, _ = data_helper.rand_arr_selection(src_data, min(300, src_data.shape[0]))
    tgt_data, _ = data_helper.rand_arr_selection(tgt_data, min(300, tgt_data.shape[0]))
    div_idx1 = src_data.shape[0]
    tsne = TSNE(n_components=2, n_iter=500).fit_transform(np.vstack([src_data, tgt_data]))
    plt.scatter(tsne[:div_idx1, 0], tsne[:div_idx1, 1], c='b', label='Source Data')
    plt.scatter(tsne[div_idx1:, 0], tsne[div_idx1:, 1], c='r', label='Target Data')

    plt.legend(loc = 'upper left')
    if title is not None:
        plt.title(title)
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)   


def main():
    for lf, name in [[global_defs.DA.A2R, 'std_A2R'], [global_defs.DA.C2R, 'std_C2R'], [global_defs.DA.P2R, 'std_P2R']]:
        print(name, 'training')
        [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(lf)
        top_dir = global_defs.mk_dir(os.path.join(global_defs.PATH_DANN_SAVING, name))
        src_hist_filename = data_helper.npfilename(os.path.join(top_dir, 'src_hist'))
        da_hist_filename = data_helper.npfilename(os.path.join(top_dir, 'da_hist'))

        model = DANN(src_dl, tgt_dl)
        model.build_graph()

        ret = model.train_eval('source',iterations=8000)

        src_hist, te_emb1, tr_emb1 = ret[0]
        da_hist,  te_emb2, tr_emb2 = ret[1]

        np.save(src_hist_filename, src_hist)
        np.save(da_hist_filename, da_hist)

        plt_hist(src_hist, da_hist, top_dir)
        visualize_da2(te_emb1[model.num_test:], te_emb1[:model.num_test], 'Visualization Before DA',
                                  os.path.join(top_dir, 'before.png'))
        visualize_da2(tr_emb2[model.num_test:], tr_emb2[:model.num_test], 'Visualization After DA',
                                  os.path.join(top_dir, 'after.png'))



if __name__=='__main__':
    main()