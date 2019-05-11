# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: Implementation of adda network.
"""
import Lglobal_defs as global_defs
import Ldata_helper as data_helper
import numpy as np
import tensorflow as tf
import os
import itertools

class ADDA:
    def __init__(self, saving_path, src_tgt_labeled_data=None, h_neurons_encoder=[1024, 196],  h_neurons_classifier=[128],
                 h_neurons_discriminator=[128], train_test_split=0.6, opt_step=0.001, l2=0.005, keep_prob=0.5):
        self._saving_path = saving_path
        src_dl, self._tgt_dl = list(src_tgt_labeled_data)
        src_dl[1] = data_helper.labels2one_hot(src_dl[1])
        self._tgt_dl[1] = data_helper.labels2one_hot(self._tgt_dl[1])
        [self._src_train_dl, self._src_test_dl] = data_helper.labeled_data_split(src_dl, train_test_split)
        self._input_dim = self._src_train_dl[0].shape[1]
        self._label_count = self._src_train_dl[1].shape[1]
        self._classifier_curr_batch_idx = 0
        self._discriminator_curr_batch_idx = 0
        self._opt_step = opt_step
        self._l2_loss = None
        self._l2_reg = l2
        self._h_neurons_encoder = h_neurons_encoder
        self._h_neurons_classifier = h_neurons_classifier
        self._h_neurons_discriminator = h_neurons_discriminator
        self.w_generator = ADDA._w_generator
        self.b_generator = ADDA._b_generator
        self._train_keep_prob = keep_prob
        self._histories_classifier = [[],[],[]] #training, test, loss
        self._histories_discriminator = [[],[],[],[]] #generator loss, discriminator loss, tgt1, src1
        self._src_encoder_scope = 'src_encoder'
        self._tgt_encoder_scope = 'tgt_encoder'
        self._classifier_scope = 'classifier'
        self._discriminator_scope = 'discriminator'
        self._path_src_nn = global_defs.mk_dir(os.path.join(self._saving_path, self._src_encoder_scope))
        self._path_tgt_nn = global_defs.mk_dir(os.path.join(self._saving_path, self._tgt_encoder_scope))
        self._path_clf_nn = global_defs.mk_dir(os.path.join(self._saving_path, self._classifier_scope))
        self._path_dsc_nn = global_defs.mk_dir(os.path.join(self._saving_path, self._discriminator_scope))
        self._discriminator_keep_prob = None
        self._discriminator_l2_loss = None

    @staticmethod
    def _gen_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def _gen_bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    @staticmethod
    def _w_generator(w_shape):
        return tf.Variable(tf.truncated_normal(w_shape,stddev=0.1))
        #return tf.random_uniform(w_shape,-0.5,0.5)

    @staticmethod
    def _b_generator(b_shape):
        return tf.random_uniform(b_shape,-0.2,0.2)

    def _add_layer(self, input, in_sz, out_sz, l2_loss, keep_prob, trainable, act_fun=None):
        #w = tf.Variable(tf.truncated_normal([in_sz, out_sz],stddev=0.1))
        w = tf.Variable(self.w_generator(w_shape=[in_sz,out_sz]),trainable=trainable)
        l2_loss += tf.nn.l2_loss(w) * self._l2_reg
        #tf.assign_add(self._l2_loss, tf.nn.l2_loss(w) * self._l2_reg)
        b = tf.Variable(self.b_generator(b_shape=[out_sz]), trainable=trainable)
        wx_plusb = tf.matmul(input, w) + b
        output = wx_plusb if act_fun is None else act_fun(wx_plusb)
        return [tf.nn.dropout(output, keep_prob), l2_loss]

    def _constuct_encoder(self, name_scope, reuse, _h_neurons, trainable):
        with tf.variable_scope(name_scope, reuse=reuse):
            _l2_loss = tf.Variable(tf.constant(0.0))
            add_layer = self._add_layer
            _keep_prob = tf.placeholder(tf.float32)
            _xi = tf.placeholder(tf.float32, [None, self._input_dim])
            h = _xi
            prev_dim = self._input_dim
            for neurons in _h_neurons[:-1]:
                h, _l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.relu, 
                                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
                prev_dim = neurons
            _yo, _l2_loss  = add_layer(h, prev_dim, _h_neurons[-1], act_fun=tf.nn.relu, 
                                       l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
            return [_yo, _l2_loss, _keep_prob, _xi]

    def _construct_src_encoder(self, reuse, trainable):
        [_yo, _l2_loss, _keep_prob, _xi] = \
            self._constuct_encoder(self._src_encoder_scope, reuse=reuse, _h_neurons=self._h_neurons_encoder, trainable=trainable)
        self._src_encoder_yo = _yo
        self._src_encoder_l2_loss = _l2_loss
        self._src_encoder_keep_prob = _keep_prob
        self._src_encoder_xi = _xi

    def _construct_tgt_encoder(self, reuse, trainable):
        [_yo, _l2_loss, _keep_prob, _xi] = \
            self._constuct_encoder(self._tgt_encoder_scope, reuse=reuse, _h_neurons=self._h_neurons_encoder, trainable=trainable)
        self._tgt_encoder_yo = _yo
        self._tgt_encoder_l2_loss = _l2_loss
        self._tgt_encoder_keep_prob = _keep_prob
        self._tgt_encoder_xi = _xi

    def _construct_classifier(self, inputs, _l2_loss, reuse, trainable):
        with tf.variable_scope(self._classifier_scope, reuse=reuse):
            _keep_prob = tf.placeholder(tf.float32)
            add_layer = self._add_layer
            h = inputs
            prev_dim = self._h_neurons_encoder[-1]
            for neurons in self._h_neurons_classifier:
                h, _l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.relu, 
                                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
                prev_dim = neurons
            _yo, _l2_loss = add_layer(h, prev_dim, self._label_count, act_fun=None, 
                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
            self._classifier_yo = _yo
            self._classifier_l2_loss = _l2_loss
            self._classifier_keep_prob = _keep_prob
            self._classifier_y = tf.placeholder(tf.float32, [None, self._label_count])
            self._classifier_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self._classifier_y, logits=_yo)) + _l2_loss
            self._classifier_trainer = tf.train.AdamOptimizer(self._opt_step).minimize(self._classifier_loss)
            #correct_prediction = tf.equal(tf.argmax(self._classifier_yo, 1), tf.argmax(self._classifier_y, 1))
            #self._classifier_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self._classifier_acc = self._softmax_eval_acc(self._classifier_yo, self._classifier_y)

    def _next_batch_classifier(self, batch_sz):
        indices = list(range(self._classifier_curr_batch_idx, self._classifier_curr_batch_idx+batch_sz))
        self._classifier_curr_batch_idx = (batch_sz + self._classifier_curr_batch_idx) % self._src_train_dl[0].shape[0]
        indices = [i-self._src_train_dl[0].shape[0] if i >= self._src_train_dl[0].shape[0] else i for i in indices]
        return [self._src_train_dl[0][indices], self._src_train_dl[1][indices]]

    def _next_batch_discriminator(self, batch_sz=64):
        indices = list(range(self._classifier_curr_batch_idx, self._classifier_curr_batch_idx+batch_sz))
        self._discriminator_curr_batch_idx = (batch_sz + self._classifier_curr_batch_idx) % \
                        (self._src_train_dl[0].shape[0] * self._tgt_dl[0].shape[0])
        indices_src = [i%self._src_train_dl[0].shape[0] for i in indices]
        indices_tgt = [i%self._tgt_dl[0].shape[0] for i in indices]
        return [[self._src_train_dl[0][indices_src], self._src_train_dl[1][indices_src]],
                [self._tgt_dl[0][indices_tgt], self._tgt_dl[1][indices_tgt]]]

    def _construct_discriminator(self, inputs, reuse, trainable):
        with tf.variable_scope(self._discriminator_scope, reuse=reuse):
            prev_dim = self._h_neurons_encoder[:-1]
            add_layer = self._add_layer
            if self._discriminator_l2_loss is None:
                self._discriminator_l2_loss = tf.Variable(tf.constant(0.0))
            if self._discriminator_keep_prob is None:
                self._discriminator_keep_prob = tf.placeholder(tf.float32)
            for neurons in self._h_neurons_discriminator:
                h, self._discriminator_l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.sigmoid, 
                        l2_loss=self._discriminator_l2_loss, keep_prob=self._discriminator_keep_prob, trainable=trainable)
                prev_dim = neurons
            _disc_o, self._discriminator_l2_loss = add_layer(h, prev_dim, 1, act_fun=tf.nn.sigmoid, 
                l2_loss=self._discriminator_l2_loss, keep_prob=self._discriminator_keep_prob, trainable=trainable)
            return _disc_o

    def _train_classifier(self, iterations=10000, batch_sz=64):
        self._construct_src_encoder(reuse=False, trainable=True)
        self._construct_classifier(self._src_encoder_yo, self._src_encoder_l2_loss, reuse=False, trainable=True)
        self._history_classifier = [[],[],[]]
        self._sess = tf.InteractiveSession()
        self._sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            data, labels = self._next_batch_classifier(batch_sz)
            if i % 2000 == 0:
                train_acc = self._classifier_acc.eval(feed_dict=
                              {self._src_encoder_xi:data, self._classifier_y:labels, self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0})
                test_acc = self._classifier_acc.eval(feed_dict=
                              {self._src_encoder_xi:self._src_test_dl[0], self._classifier_y:self._src_test_dl[1], self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0})
                loss = self._sess.run(self._classifier_loss,
                    feed_dict={self._src_encoder_xi:self._src_train_dl[0], self._classifier_y:self._src_train_dl[1], self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0})
                print('it={}'.format(i), '  train_acc=', train_acc, '  test_acc=', test_acc, '  loss=', loss)
                self._history_classifier[0].append(train_acc)
                self._history_classifier[1].append(test_acc)
                self._history_classifier[2].append(loss)
            self._classifier_trainer.run(feed_dict=
                              {self._src_encoder_xi:data, self._classifier_y:labels, self._classifier_keep_prob:self._train_keep_prob, self._src_encoder_keep_prob:self._train_keep_prob})
        self._sess.close()

    def _build_ad_loss(self):
        disc_s  = self._construct_discriminator(self._src_encoder_yo, reuse=False, trainable=True)
        disc_t = self._construct_discriminator(self._tgt_encoder_yo, reuse=True, trainable=True)
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_t, labels=tf.ones_like(disc_t))
        g_loss = tf.reduce_mean(g_loss) 
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s,labels=tf.ones_like(disc_s)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t,labels=tf.zeros_like(disc_t)))
        self._generator_loss = g_loss + self._tgt_encoder_l2_loss
        self._discriminator_loss = d_loss + self._discriminator_l2_loss
        self._tgt_disc_acc1 = self._softmax_eval_acc(logits=disc_t, labels=tf.ones_like(disc_t))
        self._src_disc_acc1 = self._softmax_eval_acc(logits=disc_s, labels=tf.ones_like(disc_s))

    def _softmax_eval_acc(self, logits, labels):
        pred = tf.nn.softmax(logits)
        #correct_label_predicted = tf.equal(tf.cast(tf.argmax(labels,axis=1),tf.int32),tf.cast(tf.argmax(pred,axis=1),tf.int32))
        correct_label_predicted = tf.equal(tf.argmax(labels,axis=1),tf.argmax(pred,axis=1))
        predicted_accuracy = tf.reduce_mean(tf.cast(correct_label_predicted,tf.float32))
        return predicted_accuracy

    def _train_discriminator(self, iterations=10000, batch_sz=64):
        self._construct_src_encoder(reuse=False, trainable=False)
        self._construct_tgt_encoder(reuse=False, trainable=True)
        self._build_ad_loss()
        tgt_encoder_train_variables = tf.trainable_variables(self._tgt_encoder_scope)
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self._opt_step,beta1=0.5,
                                                beta2=0.999).minimize(self._generator_loss,var_list=tgt_encoder_train_variables)
        disc_trainable_variables = tf.trainable_variables(self._discriminator_scope)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=self._opt_step,beta1=0.5,
                                                beta2=0.999).minimize(self._discriminator_loss,var_list=disc_trainable_variables)
        self._sess = tf.InteractiveSession()
        self._sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            [src_dl, tgt_dl] = self._next_batch_discriminator(batch_sz)
            dict = {self._tgt_encoder_xi: tgt_dl[0], self._src_encoder_xi: src_dl[0], 
                    self._tgt_encoder_keep_prob: self._train_keep_prob, self._discriminator_keep_prob: self._train_keep_prob} 
            optimizer_gen.run(dict=dict)
            optimizer_disc.run(dict=dict)
            if i % 1000 == 0:
                gen_loss = self._generator_loss.eval(dict=dict)
                disc_loss = self._discriminator_loss.eval(dict=dict)
                tgt1 = self._tgt_disc_acc1.eval(dict=dict)
                src1 = self._src_disc_acc1.eval(dict=dict)
                print("it:", i, 'gen_loss=',gen_loss,'disc_loss=',disc_loss,'target1=',tgt1, 'source1=',src1)
                self._histories_discriminator[0].append(gen_loss)
                self._histories_discriminator[1].append(disc_loss)
                self._histories_discriminator[2].append(tgt1)
                self._histories_discriminator[3].append(src1)
        self._sess.close()

    def historys(self):
        return [self._history_classifier, self._histories_discriminator]

    def neurons(self):
        return [self._input_dim] + list(self._h_neurons) + [self._label_count]

    @staticmethod
    def neuron_arrange(possible_neurons, h_layers=None, is_strict=False):
        if h_layers is None:
            h_layers = len(possible_neurons)
        if not is_strict:
            possible_neurons += possible_neurons
        neurons = sorted(possible_neurons, key=lambda x:-x)
        ret_list = []
        for perm in list(itertools.permutations(neurons, h_layers)):
            should_include = True
            for i in range(len(perm)-1):
                if perm[i+1] > perm[i]:
                    should_include = False
                    break
            if should_include:
                for p in ret_list:
                    for i in range(len(p)):
                        if perm[i] != p[i]:
                            break
                    if perm[i] == p[i]:
                        should_include = False
                        break
            if should_include:
                ret_list.append(perm)
        return ret_list

    def train(self, iterations=10000, batch_sz=50, do_close_sess=True):
        self._construct_nn()
        self._sess = tf.InteractiveSession()
        self._sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            data, labels = self._next_batch(batch_sz)
            if i % 500 == 0:
                train_acc = self._eval([data, labels])
                test_acc = self._eval(self._t_ld)
                loss = self._sess.run(self._loss, feed_dict={self._x:data, self._y:labels, self._train_keep_prob:1.0})
                print('it={}'.format(i), '  train_acc=', train_acc, '  test_acc=', test_acc, '  loss=', loss)
                self._histories[0].append(train_acc)
                self._histories[1].append(test_acc)
                self._histories[2].append(loss)
            self._trainer.run(feed_dict={self._x:data, self._y:labels, self._train_keep_prob:self._train_keep_prob})
        if do_close_sess:
            self._sess.close()


def _test_classifier():
    da_type = global_defs.DA.A2R
    dl_src, dl_tgt = data_helper.read_paired_labeled_features(da_type)
    #64 labels
    #print(dl_src)
    #neurons = GeneralNNClassifier.neuron_arrange([128, 196, 256, 512, 1024], 3, True)
    neurons = [[1024, 196, 128]]
    to_prt = []
    iterations=30000
    batch_sz = 256

    adda = ADDA(global_defs.mk_dir( os.path.join(global_defs.PATH_SAVING, 'ADDA_test')), [dl_src, dl_tgt])
    adda._train_classifier()


if __name__=='__main__':
    _test_classifier()