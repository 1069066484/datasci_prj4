# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: Implementation of adda network.
"""
import Lglobal_defs as global_defs
import Ldata_helper as data_helper
import numpy as np
import tensorflow as tf


class ADDA:
    def __init__(self, saving_path, src_tgt_labeled_data=None, h_neurons_encoder=[1024, 512],  h_neurons_classifier=[256],
                 h_neurons_discriminator=[128], train_test_split=0.6, opt_step=0.02, l2=0.5, keep_prob=0.5):
        self._saving_path = saving_path
        src_dl, self._tgt_dl = src_tgt_labeled_data
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
        self._src_encoder_scope = 'src_encoder'
        self._tgt_encoder_scope = 'tgt_encoder'
        self._classifier_scope = 'classifier'
        self._discriminator_scope = 'discriminator'

    @staticmethod
    def _gen_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def _gen_bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    @staticmethod
    def _w_generator(w_shape):
        #return tf.Variable(tf.truncated_normal([in_sz, out_sz],stddev=0.1))
        return tf.random_uniform(w_shape,-0.5,0.5)

    @staticmethod
    def _b_generator(b_shape):
        return tf.random_uniform(b_shape,-0.5,0.5)

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
                h, _l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.sigmoid, 
                                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
                prev_dim = neurons
            _yo, _l2_loss  = add_layer(h, prev_dim, _h_neurons[:-1], act_fun=tf.nn.sigmoid, 
                                       l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
            return [_yo, _l2_loss, keep_prob, _xi]

    def _construct_src_encoder(self, reuse, trainable):
        [_yo, _l2_loss, _keep_prob, _xi] = \
            self._constuct_encoder(self._src_encoder_scope, reuse=reuse, _h_neurons=self._h_neurons_encoder, trainable=trainable)
        self._src_encoder_yo = _yo
        self._src_encoder_l2_loss = _l2_loss
        self._src_encoder_keep_prob = _keep_prob
        self._src_encoder_keep_xi = _xi

    def _construct_tgt_encoder(self, reuse, trainable):
        [_yo, _l2_loss, _keep_prob, _xi] = \
            self._constuct_encoder(self._tgt_encoder_scope, reuse=reuse, _h_neurons=self._h_neurons_encoder, trainable=trainable)
        self._tgt_encoder_yo = _yo
        self._tgt_encoder_l2_loss = _l2_loss
        self._tgt_encoder_keep_prob = _keep_prob
        self._tgt_encoder_keep_xi = _xi

    def _construct_classifier(self, inputs, _l2_loss, reuse, trainable):
        with tf.variable_scope(self._classifier_scope, reuse=reuse):
            _keep_prob = tf.placeholder(tf.float32)
            add_layer = self._add_layer
            h = inputs
            prev_dim = self._h_neurons_encoder[:-1]
            for neurons in self._h_neurons_classifier:
                h, _l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.sigmoid, 
                                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
                prev_dim = neurons
            _yo, _l2_loss = add_layer(h, prev_dim, self._h_neurons_classifier[:-1], act_fun=tf.nn.sigmoid, 
                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
            self._classifier_yo = _yo
            self._classifier_l2_loss = _l2_loss
            self._classifier_keep_prob = _keep_prob
            self._classifier_y = tf.placeholder(tf.float32, [None, self._label_count])
            self._classifier_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self._classifier_y, logits=_yo)) + l2_loss
            self._classifier_trainer = tf.train.AdamOptimizer(self._opt_step).minimize(self._classifier_loss)
            correct_prediction = tf.equal(tf.argmax(self._classifier_yo, 1), tf.argmax(self._classifier_y, 1))
            self._classifier_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _next_batch_classifier(self, batch_sz):
        indices = list(range(self._classifier_curr_batch_idx, self._classifier_curr_batch_idx+batch_sz))
        self._classifier_curr_batch_idx = (batch_sz + self._classifier_curr_batch_idx) % self._data.shape[0]
        indices = [i-self._src_train_dl[0].shape[0] if i >= self._src_train_dl[0].shape[0] else i for i in indices]
        return [self._src_train_dl[0][indices], self._src_train_dl[1][indices]]

    def _construct_discriminator(self, inputs, trainable):
        with tf.variable_scope(self._discriminator_scope, reuse=reuse):
            prev_dim = self._h_neurons_encoder[:-1]
            add_layer = self._add_layer
            _l2_loss = tf.Variable(tf.constant(0.0))
            _keep_prob = tf.placeholder(tf.float32)
            for neurons in self._h_neurons_discriminator:
                h, _l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.sigmoid, 
                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
                prev_dim = neurons
            _disc_o, _l2_loss = add_layer(h, prev_dim, 1, act_fun=tf.nn.sigmoid, 
                l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable)
            return [_disc_o, _l2_loss, _keep_prob]

    def _train_classifier(self, iterations=10000, batch_sz=64):
        self._construct_src_encoder(reuse=False, trainable=True)
        self._construct_classifier(self._src_encoder_yo, self._src_encoder_l2_loss, reuse=False, trainable=True)
        self.history_classifier_training = [[],[],[]]
        self._sess = tf.InteractiveSession()
        self._sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            data, labels = self._next_batch_classifier(batch_sz)
            if i % 500 == 0:
                train_acc = self._eval([data, labels])
                test_acc = self._eval(self._t_ld)
                loss = self._sess.run(
                    self._classifier_loss, feed_dict=
                    {self._src_encoder_keep_xi:self._src_test_dl[0], self._classifier_y:self._src_test_dl[1], 
                     self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0})
                print('it={}'.format(i), '  train_acc=', train_acc, '  test_acc=', test_acc, '  loss=', loss)
                self.history_classifier_training[0].append(train_acc)
                self.history_classifier_training[1].append(test_acc)
                self.history_classifier_training[2].append(loss)
            self._classifier_trainer.run(feed_dict=
                              {self._src_encoder_keep_xi:self._src_data_dl[0], self._classifier_y:self._src_data_dl[0], self._keep_prob:self._train_keep_prob})
        if do_close_sess:
            self._sess.close()

    def _train_discriminator(self, iterations=10000, batch_sz=64):
        self._construct_src_encoder(reuse=False, trainable=False)
        [_src_disc_o, _src_l2_loss, _src_keep_prob] = self._construct_discriminator(
            self._src_encoder_yo, trainable=True)

        self._construct_tgt_encoder
        
    
    def _eval(self, labeled_data):
        return self._acc.eval(feed_dict={self._x: labeled_data[0], self._y:labeled_data[1], self._keep_prob:1.0})

    def history_classifier_training(self):
        return self.history_classifier_training

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
                loss = self._sess.run(self._loss, feed_dict={self._x:data, self._y:labels, self._keep_prob:1.0})
                print('it={}'.format(i), '  train_acc=', train_acc, '  test_acc=', test_acc, '  loss=', loss)
                self._histories[0].append(train_acc)
                self._histories[1].append(test_acc)
                self._histories[2].append(loss)
            self._trainer.run(feed_dict={self._x:data, self._y:labels, self._keep_prob:self._train_keep_prob})
        if do_close_sess:
            self._sess.close()
