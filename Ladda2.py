# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: Implementation of adda network (slim version).
            Batch normalization is supported in the codes.
"""
import tensorflow as tf
import numpy as np
import pickle as pkl
import Ldata_helper as data_helper
import Lglobal_defs as global_defs
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import os


class ADDA:
    def __init__(self, path, src_dl, tgt_dl, opt_step=0.00005, batch_size=256, keep_prob=0.95,
                 #encoder_h_layers=[1024, 500], clf_h_layers=[], da_h_layers=[500, 500],
                 encoder_h_layers=[2048], clf_h_layers=[1024], da_h_layers=[512, 512],
                 #encoder_h_layers=[1024], clf_h_layers=[256], da_h_layers=[512, 512],
                 #!encoder_h_layers=[2048], clf_h_layers=[1024], da_h_layers=[512, 512],
                 #!encoder_h_layers=[2048], clf_h_layers=[], da_h_layers=[512, 512],
                 #!encoder_h_layers=[4096], clf_h_layers=[], da_h_layers=[512, 512],
                 l2=1e-5, use_bn=False):
        tf.reset_default_graph()
        self.path = path
        self.use_bn = use_bn
        self.opt_step = opt_step
        self.l2 = l2

        src_dl[1] = data_helper.labels2one_hot(src_dl[1])
        tgt_dl[1] = data_helper.labels2one_hot(tgt_dl[1])
        
        self.tgt_tr_dl, self.tgt_te_dl = data_helper.labeled_data_split(tgt_dl, 0.6)
        self.src_tr_dl, self.src_te_dl = data_helper.labeled_data_split(src_dl, 0.9)

        self.input_dim_tgt = self.tgt_tr_dl[0].shape[1]
        self.input_dim_src = self.src_tr_dl[0].shape[1]
        self.label_count = self.src_tr_dl[1].shape[1]

        self.batch_size = batch_size

        self.encoder_h_layers = encoder_h_layers
        self.clf_h_layers = clf_h_layers
        self.da_h_layers = da_h_layers

        self.dsc_curr_batch_idx = 0
        self.clf_curr_batch_idx = 0

        self.kp = keep_prob
        self._build_model()

    def _add_layer(self, input, out_dim, act_fun=tf.nn.relu):
        input_dim = input.shape.as_list()[1]
        w = weight_variable([input_dim, out_dim])
        b = bias_variable([out_dim])
        raw_output = tf.matmul(input, w) + b
        return raw_output if act_fun is None else act_fun(raw_output)

    def fc(self, input, layer, scope, layer_idx, act_fun=tf.nn.leaky_relu):
        return slim.fully_connected(input,layer,scope=scope + str(layer_idx),
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    biases_initializer=tf.random_uniform_initializer(-0.1,0.1),
                                    normalizer_fn=slim.batch_norm if self.use_bn else None,
                                    weights_regularizer=slim.l2_regularizer(self.l2),
                                    #activation_fn=tf.nn.sigmoid
                                    activation_fn=tf.nn.relu
                                    #activation_fn=tf.nn.leaky_relu
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

        self.lr = tf.Variable(self.opt_step * 0.1, name='lr', trainable=False)

        batch_size = self.batch_size
        
        with tf.variable_scope("src_ec"):
            net = self.src_x
            for idx, layer in enumerate(self.encoder_h_layers):
                #net=slim.fully_connected(net,layer,scope="src_ec" + str(idx))
                net = self.fc(net, layer, 'src_ec', idx)
                net=slim.dropout(net,is_training=self.src_is_tr,scope="src_ec_drop" + str(idx), keep_prob=self.kp)
            self.src_ec_yo = net

        with tf.variable_scope("tgt_ec"):
            net = self.tgt_x
            for idx, layer in enumerate(self.encoder_h_layers):
                net = self.fc(net, layer, 'tgt_ec', idx)
                net=slim.dropout(net,is_training=self.tgt_is_tr,scope="tgt_ec_drop" + str(idx), keep_prob=self.kp)
            self.tgt_ec_yo = net

        with tf.variable_scope("clf"):
            net = tf.cond(self.clf_use_src, lambda: self.src_ec_yo, lambda: self.tgt_ec_yo)
            for idx, layer in  enumerate(self.clf_h_layers):
                net = self.fc(net, layer, 'clf', idx)
                net = slim.dropout(net,is_training=self.clf_is_tr,scope="clf_drop" + str(idx), keep_prob=self.kp)
            clf_yo = self.fc(net, self.label_count, 'clf_out', 0, None)
            la = tf.losses.get_regularization_losses()
            self.clf_loss = tf.add_n( tf.losses.get_regularization_losses() +
                                       [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                           labels=self.clf_y, logits=clf_yo))])
            clf_pred = tf.equal(tf.argmax(self.clf_y,axis=1),tf.argmax(clf_yo,axis=1))
            self.clf_acc = tf.reduce_mean(tf.cast(clf_pred,tf.float32))
           
        with tf.variable_scope("dsc"):
            net = tf.concat([self.src_ec_yo, self.tgt_ec_yo], 0)
            labels = tf.concat( [tf.zeros([tf.shape(self.src_ec_yo)[0]], tf.int64),  
                                 tf.ones([tf.shape(self.tgt_ec_yo)[0]], tf.int64)], 0)
            for idx, layer in enumerate(self.da_h_layers):
                net = self.fc(net, layer, 'dsc', idx)
                net=slim.dropout(net,is_training=self.dsc_is_tr,scope="dsc_drop" + str(idx), keep_prob=self.kp)
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

        
    def train(self, iterations=[8000, 12000], do_preassign=True, do_dec_lr=True, dsc_quit_drop=3.0):
        if isinstance(iterations, int):
            iterations = [iterations, round(iterations*1.0)]
        optimizer_clf = tf.train.AdamOptimizer(self.opt_step,beta1=0.5,beta2=0.9).minimize(
            self.clf_loss, var_list=tf.global_variables('clf') + tf.global_variables('src_ec'))
        optimizer_dsc = tf.train.AdamOptimizer(self.opt_step,beta1=0.5,beta2=0.9).minimize(
            self.dsc_loss, var_list=tf.global_variables('dsc'))
        optimizer_tgt = tf.train.AdamOptimizer(self.lr,beta1=0.5,beta2=0.9).minimize(
            self.tgt_ec_loss, var_list=tf.global_variables('tgt_ec'))

        with tf.Session() as self._sess:
            self._sess.run(tf.global_variables_initializer())
            ###########################################################################################
            #######################Train Source Encoder And Classifier#################################
            ###########################################################################################
            clf_hist = []
            true_dict = {self.src_is_tr: True, self.tgt_is_tr: True, 
                            self.clf_is_tr: True, self.dsc_is_tr: True}
            for i in range(iterations[0]):
                if do_dec_lr and i % 100 == 0:
                    self._sess.run(self.lr.assign(self.opt_step * 1.5 / (1 + i / 100.0) ))
                else:
                    self._sess.run(self.lr.assign(self.opt_step * 1.0 ))
                clf_x, clf_y = self._next_batch_clf()
                clf_dict = {self.src_x: clf_x, self.clf_y: clf_y, self.src_is_tr: True, 
                            self.tgt_x: np.zeros([1,int(self.tgt_x.shape[1])]),
                            self.tgt_is_tr: False, self.clf_is_tr: True, self.clf_use_src: True,
                            self.dsc_is_tr: False}
                optimizer_clf.run(feed_dict=clf_dict)
                if i % 500 == 0 or i == iterations[0]-1:
                    clf_dict.update(true_dict)
                    train_acc = self.clf_acc.eval(feed_dict=clf_dict)
                    loss = self.clf_loss.eval(feed_dict=clf_dict)
                    clf_dict.update({self.src_x: self.src_te_dl[0], self.clf_y: self.src_te_dl[1]})
                    test_acc = self.clf_acc.eval(feed_dict=clf_dict)
                    print("Classifier",i, loss, train_acc, test_acc)
                    clf_hist.append([i, loss, train_acc, test_acc])
            self.clf_hist = np.array(clf_hist)
            src_ec = self.src_ec_yo.eval(feed_dict=clf_dict)

            ###########################################################################################
            ####################Copy Encoder Parameters(Pretrain Target Encoder)#######################
            ###########################################################################################
            if do_preassign:
                for src_v in tf.global_variables('src_ec'):
                    for tgt_v in tf.global_variables('tgt_ec'):
                        if src_v.name[11:] == tgt_v.name[11:]:
                            #src_v = self._sess.run(src_v)
                            self._sess.run(tf.assign(tgt_v, src_v))

            ###########################################################################################
            #######################Train Target Encoder And Discriminator##############################
            ###########################################################################################
            gan_hist = []
            for i in range(iterations[1]):
                if do_dec_lr and i % 100 == 0:
                    self._sess.run(self.lr.assign(self.opt_step * 0.1 / (1 + i / 100.0) ))
                else:
                    self._sess.run(self.lr.assign(self.opt_step * 0.05 ))
                src_dl, tgt_dl = self._next_batch_dsc()
                dsc_dict = {self.src_x: src_dl[0], self.clf_y: np.zeros([1,int(self.clf_y.shape[1])]), 
                            self.src_is_tr: False, self.tgt_x: tgt_dl[0], self.tgt_is_tr: False, 
                            self.clf_is_tr: False, self.clf_use_src: False, self.dsc_is_tr: True}

                optimizer_dsc.run(feed_dict=dsc_dict)
                tgt_dict = {self.src_x: src_dl[0], self.clf_y: np.zeros([1,int(self.clf_y.shape[1])]), 
                            self.src_is_tr: False, self.tgt_x: tgt_dl[0], self.tgt_is_tr: True, 
                            self.clf_is_tr: False, self.clf_use_src: False, self.dsc_is_tr: False}
                optimizer_tgt.run(feed_dict=dsc_dict)
                if i % 500 == 0 or i == iterations[1]-1:
                    tgt_dict.update(true_dict)
                    dsc_dict.update(true_dict)
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

                    if i == 0:
                        tgt_ec0 = self.tgt_ec_yo.eval(feed_dict=clf_dict)

                        clf_dict.update({self.src_x: self.tgt_tr_dl[0], self.clf_y:self.tgt_tr_dl[1], 
                                            self.clf_use_src:True})
                        src_clf_tr_acc = self.clf_acc.eval(feed_dict=clf_dict)

                        clf_dict.update({self.src_x: self.tgt_te_dl[0], self.clf_y:self.tgt_te_dl[1], 
                                            self.clf_use_src:True})
                        src_clf_te_acc = self.clf_acc.eval(feed_dict=clf_dict)
  
                    print('GAN', i, dsc_loss, tgt_loss, dsc_tr_acc, dsc_te_acc, tgt_clf_tr_acc, 
                          tgt_clf_te_acc, src_clf_tr_acc, src_clf_te_acc)
                    gan_hist.append([i, dsc_loss, tgt_loss, dsc_tr_acc, dsc_te_acc, tgt_clf_tr_acc, 
                          tgt_clf_te_acc, src_clf_tr_acc, src_clf_te_acc])
                    if dsc_quit_drop is not None and dsc_quit_drop < src_clf_te_acc - tgt_clf_te_acc:
                        break

            clf_dict.update({self.clf_use_src:False})
            tgt_ec1 = self.tgt_ec_yo.eval(feed_dict=clf_dict)
            self.gan_hist = np.array(gan_hist)
            data_helper.visualize_da(src_ec, 
                                     tgt_ec0, 
                                     tgt_ec1,
                                     'ADDA Transfer Visualization',
                                     os.path.join(self.path, 'ADDA_visual.png'))
            self._save_params(iterations, do_preassign, do_dec_lr, dsc_quit_drop)
            self.clf_plt()
            self.gan_plt()

    def _save_params(self,iterations, do_preassign, do_dec_lr, dsc_quit_drop):
        np.save(os.path.join(self.path, data_helper.npfilename('clf_hist')), self.clf_hist)
        np.save(os.path.join(self.path, data_helper.npfilename('gan_hist')), self.gan_hist)
        hyper_params = [self.opt_step, self.l2, self.batch_size, self.encoder_h_layers, self.clf_h_layers, self.da_h_layers,
                        self.kp, iterations, do_preassign, do_dec_lr, dsc_quit_drop, self.use_bn]
        print(hyper_params)
        pkl.dump(hyper_params, open(os.path.join(self.path, data_helper.pkfilename('hyper_params')), 'wb'))

    def clf_plt(self, clf_hist=None):
        plt.figure(figsize=(15,10))
        if clf_hist is None:
            clf_hist = self.clf_hist
        clf_hist[:, 1] -= np.min(clf_hist[:, 1]); clf_hist[:, 1] /= np.max(clf_hist[:, 1])
        plt.plot(clf_hist[:, 0], clf_hist[:, 1], label='loss')
        plt.plot(clf_hist[:, 0], clf_hist[:, 2], label='training accuracy')
        plt.plot(clf_hist[:, 0], clf_hist[:, 3], label='test accuracy')
        plt.legend(['classifier loss', 'training accuracy', 'test accuracy'])
        plt.xlabel('Iterations')
        plt.title('ADDA source encoder and classifier training history')
        #plt.show()
        plt.savefig(os.path.join(self.path, 'clf.png'))

    def gan_plt(self, gan_hist=None):
        if gan_hist is None:
            gan_hist = self.gan_hist
        plt.figure(figsize=(25,10))
        plt.subplot(121)
        gan_hist[:, 1] -= np.min(gan_hist[:, 1]); gan_hist[:, 1] /= np.max(gan_hist[:, 1])
        gan_hist[:, 2] -= np.min(gan_hist[:, 2]); gan_hist[:, 2] /= np.max(gan_hist[:, 2])
        plt.plot(gan_hist[:, 0], gan_hist[:, 1], label='discriminator loss')
        plt.plot(gan_hist[:, 0], gan_hist[:, 2], label='target encoder loss')
        plt.plot(gan_hist[:, 0], gan_hist[:, 3], label='target training set acc')
        plt.plot(gan_hist[:, 0], gan_hist[:, 4], label='target test set acc')
        plt.legend(['discriminator loss', 'target encoder loss', 'target training set acc', 'target test set acc'])
        plt.xlabel('Iterations')
        plt.title('ADDA GAN training history of loss and discriminator')
        #plt.show()
    
        plt.subplot(122)
        plt.plot(gan_hist[:, 0], gan_hist[:, 5], label='acc(t_ec, training set)')
        plt.plot(gan_hist[:, 0], gan_hist[:, 6], label='acc(t_ec, test set)')

        plt.plot(gan_hist[:, 0], gan_hist[:, 7], label='acc(s_ec, training set)')
        plt.plot(gan_hist[:, 0], gan_hist[:, 8], label='acc(s_ec, test set)')
        plt.legend(['acc(t_ec, training set)', 'acc(t_ec, test set)', 'acc(s_ec, training set)', 'acc(s_ec, test set)'])
        plt.xlabel('Iterations')
        plt.title('ADDA GAN training history of accuracies')
        plt.savefig(os.path.join(self.path, 'gan.png'))


iterations = 8000
#no_pre_as, no_dec, no_bn -> no_dec, no_bn -> no_bn -> bn
def main1():
    [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(global_defs.DA.A2R)
    path = global_defs.mk_dir(os.path.join(global_defs.PATH_ADDA_SAVING, 'A2R_no_preassign_no_dec_no_bn'))
    adda = ADDA(path, src_dl, tgt_dl, use_bn=False)
    adda.train(iterations=iterations, do_preassign=False, do_dec_lr=False)

def main2():
    [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(global_defs.DA.A2R)
    path = global_defs.mk_dir(os.path.join(global_defs.PATH_ADDA_SAVING, 'A2R_no_dec_no_bn'))
    adda = ADDA(path, src_dl, tgt_dl, use_bn=False)
    adda.train(iterations=iterations, do_preassign=True, do_dec_lr=False)

def main3():
    [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(global_defs.DA.A2R)
    path = global_defs.mk_dir(os.path.join(global_defs.PATH_ADDA_SAVING, 'A2R_no_bn'))
    adda = ADDA(path, src_dl, tgt_dl, use_bn=False)
    adda.train(iterations=iterations, do_preassign=True, do_dec_lr=True)

def main4():
    [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(global_defs.DA.A2R)
    path = global_defs.mk_dir(os.path.join(global_defs.PATH_ADDA_SAVING, 'std_A2R'))
    adda = ADDA(path, src_dl, tgt_dl, use_bn=True)
    adda.train(iterations=iterations, do_preassign=True, do_dec_lr=True)

def main5():
    [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(global_defs.DA.C2R)
    path = global_defs.mk_dir(os.path.join(global_defs.PATH_ADDA_SAVING, 'std_C2R'))
    adda = ADDA(path, src_dl, tgt_dl, use_bn=True)
    adda.train(iterations=iterations)

def main6():
    [src_dl, tgt_dl] = data_helper.read_paired_labeled_features(global_defs.DA.P2R)
    path = global_defs.mk_dir(os.path.join(global_defs.PATH_ADDA_SAVING, 'std_P2R'))
    adda = ADDA(path, src_dl, tgt_dl, use_bn=True)
    adda.train(iterations=iterations)

if __name__=='__main__':
    for f in [
        main1,
        main2,
        main3,
        main4,
        main5,
        main6
        ]:
        print(f)
        f()
