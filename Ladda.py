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
from enum import IntEnum


class ADDA:
    Classifier = 1
    Discriminator = 2
    DomainAdapt = 4

    def __init__(self, saving_path, src_tgt_labeled_data=None, h_neurons_encoder=[1024, 196],  h_neurons_classifier=[128],
                 h_neurons_discriminator=[128], train_test_split=0.6, opt_step=0.001, l2=0.005, keep_prob=0.5):
        #print(ADDA.Classifier)
        #exit(1)
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

    def _add_layer(self, input, in_sz, out_sz, l2_loss, keep_prob, trainable, act_fun=None, layer_idx=None):
        #w = tf.Variable(tf.truncated_normal([in_sz, out_sz],stddev=0.1))
        w = tf.get_variable(initializer=self.w_generator(w_shape=[in_sz,out_sz]),trainable=trainable, name=None if layer_idx is None else 'w'+str(layer_idx))
        l2_loss += tf.nn.l2_loss(w) * self._l2_reg
        #tf.assign_add(self._l2_loss, tf.nn.l2_loss(w) * self._l2_reg)
        b = tf.get_variable(initializer=self.b_generator(b_shape=[out_sz]), trainable=trainable, name=None if layer_idx is None else 'b'+str(layer_idx))
        wx_plusb = tf.matmul(input, w) + b
        output = wx_plusb if act_fun is None else act_fun(wx_plusb)
        return [tf.nn.dropout(output, keep_prob), l2_loss]

    def _constuct_encoder(self, name_scope, reuse, _h_neurons, trainable):
        with tf.variable_scope(name_scope, reuse=reuse):
            _l2_loss = tf.Variable(tf.constant(0.0), name='l2_loss')
            add_layer = self._add_layer
            _keep_prob = tf.placeholder(tf.float32)
            _xi = tf.placeholder(tf.float32, [None, self._input_dim])
            h = _xi
            prev_dim = self._input_dim
            layer_idx = 0
            for neurons in _h_neurons[:-1]:
                h, _l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.relu, 
                                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable, layer_idx=layer_idx)
                prev_dim = neurons
                layer_idx += 1
            _yo, _l2_loss  = add_layer(h, prev_dim, _h_neurons[-1], act_fun=tf.nn.relu, 
                                       l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable, layer_idx=layer_idx)
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
            layer_idx = 0
            for neurons in self._h_neurons_classifier:
                h, _l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.relu, 
                                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable, layer_idx=layer_idx)
                prev_dim = neurons
                layer_idx += 1
            _yo, _l2_loss = add_layer(h, prev_dim, self._label_count, act_fun=None, 
                        l2_loss=_l2_loss, keep_prob=_keep_prob, trainable=trainable, layer_idx=layer_idx)
            self._classifier_yo = _yo
            self._classifier_l2_loss = _l2_loss
            self._classifier_keep_prob = _keep_prob
            self._classifier_y = tf.placeholder(tf.float32, [None, self._label_count])
            self._classifier_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self._classifier_y, logits=_yo)) + _l2_loss
            #correct_prediction = tf.equal(tf.argmax(self._classifier_yo, 1), tf.argmax(self._classifier_y, 1))
            #self._classifier_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self._classifier_acc = self._softmax_eval_acc(self._classifier_yo, self._classifier_y)

    def _next_batch_classifier(self, batch_sz):
        indices = list(range(self._classifier_curr_batch_idx, self._classifier_curr_batch_idx+batch_sz))
        self._classifier_curr_batch_idx = (batch_sz + self._classifier_curr_batch_idx) % self._src_train_dl[0].shape[0]
        indices = [i-self._src_train_dl[0].shape[0] if i >= self._src_train_dl[0].shape[0] else i for i in indices]
        return [self._src_train_dl[0][indices], self._src_train_dl[1][indices]]

    def _next_batch_discriminator(self, batch_sz=64):
        indices = list(range(self._discriminator_curr_batch_idx, self._discriminator_curr_batch_idx+batch_sz))
        self._discriminator_curr_batch_idx = (batch_sz + self._discriminator_curr_batch_idx) % \
                        (self._src_train_dl[0].shape[0] * self._tgt_dl[0].shape[0])
        indices_src = [i%self._src_train_dl[0].shape[0] for i in indices]
        indices_tgt = [i%self._tgt_dl[0].shape[0] for i in indices]

        return [[self._src_train_dl[0][indices_src], self._src_train_dl[1][indices_src]],
                [self._tgt_dl[0][indices_tgt], self._tgt_dl[1][indices_tgt]]]

    def _construct_discriminator(self, inputs, reuse, trainable):
        with tf.variable_scope(self._discriminator_scope, reuse=reuse):
            prev_dim = self._h_neurons_encoder[-1]
            layer_idx = 0
            add_layer = self._add_layer
            if self._discriminator_l2_loss is None:
                self._discriminator_l2_loss = tf.get_variable(initializer=tf.constant(0.0), name="l2_loss")
            if self._discriminator_keep_prob is None:
                self._discriminator_keep_prob = tf.placeholder(tf.float32)
            h = inputs
            for neurons in self._h_neurons_discriminator:
                h, self._discriminator_l2_loss = add_layer(h, prev_dim, neurons, act_fun=tf.nn.sigmoid, layer_idx=layer_idx,
                        l2_loss=self._discriminator_l2_loss, keep_prob=self._discriminator_keep_prob, trainable=trainable)
                prev_dim = neurons
                layer_idx += 1
            _disc_o, self._discriminator_l2_loss = add_layer(h, prev_dim, 2, act_fun=tf.nn.sigmoid,  layer_idx=layer_idx,
                l2_loss=self._discriminator_l2_loss, keep_prob=self._discriminator_keep_prob, trainable=trainable)
            return _disc_o

    def _try_get_saver(self, scope, path, do_clear, do_raise=None, ret_state=False, only_trainable=True):
        #vars = tf.trainable_variables(scope=scope) if only_trainable else tf.global_variables(scope=scope)
        vars = tf.global_variables(scope=scope)
        saver = tf.train.Saver(var_list=vars)
        ckpt = tf.train.get_checkpoint_state(path)
        success = False
        if not do_clear and ckpt and ckpt.model_checkpoint_path:
            print("{} check point is found!".format(scope))
            saver.restore(self._sess, ckpt.model_checkpoint_path)
            success = True
        elif do_raise is not None:
            raise Exception(do_raise)
        return saver if not ret_state else [saver, success]

    def _save_nn(self, saver, path):
        saver.save(self._sess, os.path.join(path, 'model.ckpt'))

    def _train_classifier(self, iterations=10000, batch_sz=64, do_clear=False):
        tf.reset_default_graph()
        self._construct_tgt_encoder(reuse=False, trainable=False)
        self._construct_classifier(self._tgt_encoder_yo, self._tgt_encoder_l2_loss, reuse=False, trainable=True)
        self._construct_src_encoder(reuse=False, trainable=True)
        self._construct_classifier(self._src_encoder_yo, self._src_encoder_l2_loss, reuse=True, trainable=True)
        self._classifier_trainer = tf.train.AdamOptimizer(self._opt_step).minimize(self._classifier_loss)
        self._history_classifier = [[],[],[]]

        with tf.Session() as self._sess:
            
            self._sess.run(tf.global_variables_initializer())
            best_acc = -1
            saver_clf = self._try_get_saver(self._classifier_scope, self._path_clf_nn, do_clear)
            saver_src_encoder = self._try_get_saver(self._src_encoder_scope, self._path_src_nn, do_clear)
            saver_tgt_encoder = self._try_get_saver(self._tgt_encoder_scope, self._path_tgt_nn, do_clear)

            for i in range(iterations):
                data, labels = self._next_batch_classifier(batch_sz)
                if i % 500 == 0:
                    train_acc = self._classifier_acc.eval(feed_dict=
                                  {self._src_encoder_xi:data, self._classifier_y:labels, self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0})
                    test_acc = self._classifier_acc.eval(feed_dict=
                                  {self._src_encoder_xi:self._src_test_dl[0], self._classifier_y:self._src_test_dl[1], self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0})
                    loss = self._sess.run(self._classifier_loss,
                        feed_dict={self._src_encoder_xi:self._src_train_dl[0], self._classifier_y:self._src_train_dl[1], self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0})
                    if test_acc > best_acc:
                        best_acc = test_acc
                        #self._copy_classifier()
                        self._save_nn(saver_clf, self._path_clf_nn)
                        self._save_nn(saver_src_encoder, self._path_src_nn)
                        #saver_clf.save(self._sess, os.path.join(self._path_clf_nn, 'model.ckpt'))
                    print('it={}'.format(i), '  train_acc=', train_acc, '  test_acc=', test_acc, '  loss=', loss, 'best_acc=',best_acc)
                    self._history_classifier[0].append(train_acc)
                    self._history_classifier[1].append(test_acc)
                    self._history_classifier[2].append(loss)
                self._classifier_trainer.run(feed_dict=
                                  {self._src_encoder_xi:data, self._classifier_y:labels, self._classifier_keep_prob:self._train_keep_prob, self._src_encoder_keep_prob:self._train_keep_prob})
            self._save_nn(saver_tgt_encoder, self._path_tgt_nn)

    def _build_ad_los2s(self):
        disc_s  = self._construct_discriminator(self._src_encoder_yo, reuse=False, trainable=True)
        disc_t = self._construct_discriminator(self._tgt_encoder_yo, reuse=True, trainable=True)
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_t, labels=tf.ones_like(disc_t))
        g_loss = tf.reduce_mean(g_loss) 
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s,labels=tf.ones_like(disc_s)))+ \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t,labels=tf.zeros_like(disc_t)))
        d_loss = -d_loss;
        self._generator_loss = g_loss + self._tgt_encoder_l2_loss
        self._discriminator_loss = d_loss + self._discriminator_l2_loss
        self._tgt_disc_acc1 = self._softmax_eval_acc(logits=disc_t, labels=tf.ones_like(disc_t))
        self._src_disc_acc1 = self._softmax_eval_acc(logits=disc_s, labels=tf.ones_like(disc_s))

    def _build_ad_loss(self):
        #disc_s  = self._construct_discriminator(self._src_encoder_yo, reuse=False, trainable=True)
        #disc_t = self._construct_discriminator(self._tgt_encoder_yo, reuse=True, trainable=True)
        source_ft = self._src_encoder_yo
        target_ft = self._tgt_encoder_yo

        adversary_ft = tf.concat([
            tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
            , tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])
            ], 0)

        source_adversary_label = tf.zeros([tf.shape(source_ft)[0]], tf.int32)
        target_adversary_label = tf.ones([tf.shape(target_ft)[0]], tf.int32)

        adversary_label = tf.concat(
            [source_adversary_label, target_adversary_label], 0)

        adversary_logits = self._construct_discriminator(adversary_ft, reuse=False, trainable=True)

        # losses
        mapping_loss = tf.losses.sparse_softmax_cross_entropy(
             1-adversary_label, adversary_logits)

        adversary_loss = tf.losses.sparse_softmax_cross_entropy(
            adversary_label, adversary_logits)

        self._generator_loss = mapping_loss + self._tgt_encoder_l2_loss * 0
        self._discriminator_loss = adversary_loss + self._discriminator_l2_loss * 0
        #self._tgt_disc_acc1 = self._softmax_eval_acc(logits=disc_t, labels=tf.ones_like(disc_t))
        #self._src_disc_acc1 = self._softmax_eval_acc(logits=disc_s, labels=tf.ones_like(disc_s))
        #self.adversary_label=adversary_label
        #.adversary_logits=adversary_logits


    def _softmax_eval_acc(self, logits, labels):
        pred = tf.nn.softmax(logits)
        #correct_label_predicted = tf.equal(tf.cast(tf.argmax(labels,axis=1),tf.int32),tf.cast(tf.argmax(pred,axis=1),tf.int32))
        correct_label_predicted = tf.equal(tf.argmax(labels,axis=1),tf.argmax(pred,axis=1))
        predicted_accuracy = tf.reduce_mean(tf.cast(correct_label_predicted,tf.float32))
        return predicted_accuracy

    def _copy_encoder(self):
        src_vars = tf.global_variables(scope=self._src_encoder_scope)
        tgt_vars = tf.global_variables(scope=self._tgt_encoder_scope)
        for src_v in src_vars:
            for tgt_v in tgt_vars:
                if src_v.name[4:] == tgt_v.name[4:]:
                    #print(src_v.name, tgt_v.name)
                    src_var_val = self._sess.run(src_v)
                    self._sess.run(tgt_v.assign(src_var_val))

    def _copy_classifier(self):
        clf_vars = tf.global_variables(scope=self._classifier_scope)
        print("len(clf_vars)=",len(clf_vars))
        len_half = round(len(clf_vars)/2)
        for i in range( len_half):
            var0 = self._sess.run(clf_vars[i+len_half])
            self._sess.run(clf_vars[i].assign(var0))

    def _train_discriminator(self, iterations=3000, batch_sz=64, do_clear=False):
        tf.reset_default_graph()
        self._construct_tgt_encoder(reuse=False, trainable=True)
        self._construct_src_encoder(reuse=False, trainable=False)
        
        self._build_ad_loss()

        with tf.Session() as self._sess:

            #self._check_vars(self._src_encoder_scope)
            #self._check_vars(self._tgt_encoder_scope)
            #self._check_vars(self._discriminator_scope, True)
            #exit(0)

            writer = tf.summary.FileWriter(os.path.join(self._saving_path,'discriminator/log'),self._sess.graph)

            tgt_encoder_train_variables = tf.trainable_variables(self._tgt_encoder_scope)
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=self._opt_step,beta1=0.5,
                                                    beta2=0.999).minimize(self._generator_loss,var_list=tgt_encoder_train_variables)
            disc_trainable_variables = tf.trainable_variables(self._discriminator_scope)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=self._opt_step,beta1=0.5,
                                                    beta2=0.999).minimize(self._discriminator_loss,var_list=disc_trainable_variables)

            self._sess.run(tf.global_variables_initializer())

            saver_src_encoder = self._try_get_saver(self._src_encoder_scope, self._path_src_nn,  False, 
                                                    "Cannot find source encoder parameters", only_trainable=False)
            saver_tgt_encoder, loaded = self._try_get_saver(self._tgt_encoder_scope, self._path_tgt_nn, do_clear, ret_state=True, 
                                                            only_trainable=False)
       
            saver_discriminator = self._try_get_saver(self._discriminator_scope, self._path_dsc_nn, do_clear)
            

            #self._check_vars(self._src_encoder_scope, True)
            #self._check_vars(self._tgt_encoder_scope, True)
            #return None
            if not loaded:
                print("_copy_encoder")
                self._copy_encoder()
            #self._check_vars(self._src_encoder_scope, True)
            #self._check_vars(self._tgt_encoder_scope, True)
            #self._save_nn(saver_tgt_encoder, self._path_tgt_nn)
            #return None
            for i in range(iterations):
                [src_dl, tgt_dl] = self._next_batch_discriminator(batch_sz)
                dict = {self._tgt_encoder_xi: tgt_dl[0], self._src_encoder_xi: src_dl[0], 
                        self._tgt_encoder_keep_prob: self._train_keep_prob, self._discriminator_keep_prob: 1.0, self._src_encoder_keep_prob: 1.0} 
                optimizer_gen.run(feed_dict=dict)
                dict[self._tgt_encoder_keep_prob] = 1.0
                dict[self._discriminator_keep_prob] = self._train_keep_prob
                optimizer_disc.run(feed_dict=dict)
                if i % 100 == 0:
                    dict[self._discriminator_keep_prob] = 1.0

                    #self._copy_encoder()
                    #print(" self._tgt_encoder_yo=",self._sess.run( self._tgt_encoder_yo, feed_dict=dict))
                   

                    #print("self._tgt_encoder_xi=",self._sess.run(self._tgt_encoder_xi, feed_dict=dict))

                    #print("self.disc_s=",self._sess.run(self.disc_s, feed_dict=dict))
                    #print("self.disc_t=",self._sess.run(self.disc_t, feed_dict=dict))
                    #print("self.adversary_label=", self._sess.run(self.adversary_label, feed_dict=dict))
                    #print("self.adversary_logits=", self._sess.run(self.adversary_logits, feed_dict=dict))

                    gen_loss = self._generator_loss.eval(feed_dict=dict)
                    disc_loss = self._discriminator_loss.eval(feed_dict=dict)
                    #tgt1 = self._tgt_disc_acc1.eval(feed_dict=dict)
                    #src1 = self._src_disc_acc1.eval(feed_dict=dict)
                    print("it:", i, 'gen_loss=',gen_loss,'disc_loss=',disc_loss,'target1=' #,tgt1, 'source1=',src1
                          )
                    self._histories_discriminator[0].append(gen_loss)
                    self._histories_discriminator[1].append(disc_loss)
                    #self._histories_discriminator[2].append(tgt1)
                    #self._histories_discriminator[3].append(src1)
            self._save_nn(saver_tgt_encoder, self._path_tgt_nn)
            self._save_nn(saver_discriminator, self._path_dsc_nn)

    def _domain_adaptation(self):
        tf.reset_default_graph()
        with tf.Session() as self._sess:
            self._construct_src_encoder(reuse=False, trainable=False)
            self._construct_classifier(self._src_encoder_yo, self._src_encoder_l2_loss, reuse=False, trainable=False)
            src_encoder_acc = self._classifier_acc
            src_encoder_feed_dict={self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0}
            src_encoder_classifier_y = self._classifier_y

            self._construct_tgt_encoder(reuse=False, trainable=False)
            self._construct_classifier(self._tgt_encoder_yo, self._tgt_encoder_l2_loss, reuse=tf.AUTO_REUSE, trainable=False)
            tgt_encoder_acc = self._classifier_acc
            tgt_encoder_feed_dict={self._classifier_keep_prob:1.0, self._tgt_encoder_keep_prob:1.0}
            tgt_encoder_classifier_y = self._classifier_y
        
            self._sess.run(tf.global_variables_initializer())

            #self._check_vars(self._src_encoder_scope, True)
            #self._check_vars(self._tgt_encoder_scope, True)

            #self._check_vars(self._classifier_scope)

            #exit(0)

            saver_src_encoder = self._try_get_saver(self._src_encoder_scope, self._path_src_nn,  False, 
                                                    "Cannot find source encoder parameters")
            saver_clf = self._try_get_saver(self._classifier_scope, self._path_clf_nn,  False, 
                                                    "Cannot find classifier parameters")
            saver_tgt_encoder = self._try_get_saver(self._tgt_encoder_scope, self._path_tgt_nn,  False, 
                                                    "Cannot find target encoder parameters")
            #self._copy_encoder()

            #self._check_vars(self._src_encoder_scope, True)
            #self._check_vars(self._tgt_encoder_scope, True)
            #self._copy_classifier()

            src_encoder_feed_dict.update({self._src_encoder_xi:self._src_test_dl[0], src_encoder_classifier_y:self._src_test_dl[1]})
            test_acc_src = src_encoder_acc.eval(feed_dict=src_encoder_feed_dict)
            src_encoder_feed_dict.update({self._src_encoder_xi:self._tgt_dl[0], src_encoder_classifier_y:self._tgt_dl[1]})
            test_acc_tgt = src_encoder_acc.eval(feed_dict=src_encoder_feed_dict)

            tgt_encoder_feed_dict.update({self._tgt_encoder_xi:self._tgt_dl[0], tgt_encoder_classifier_y:self._tgt_dl[1]})
            test_acc_tgt_adapted = tgt_encoder_acc.eval(feed_dict=tgt_encoder_feed_dict)
            print("Source domain test dataset acc:", test_acc_src, '\n',
                  "Target domain data set using source encoder acc:", test_acc_tgt, '\n',
                  "Target domain data set using target encoder acc:", test_acc_tgt_adapted, '\n')

    def train(self, train_net = Classifier | Discriminator | DomainAdapt,
              iterations = 20000, batch_sz=64, do_clear=False):

        if train_net & ADDA.Classifier != 0:
            print("\n\nClassifier Training Start")
            self._train_classifier(iterations, batch_sz, do_clear)
        if train_net & ADDA.Discriminator != 0:
            print("\n\nDiscriminator Training Start")
            self._train_discriminator(iterations, batch_sz, do_clear)
        if train_net & ADDA.DomainAdapt != 0:
            print("\n\nDomain Adaptation Start")
            self._domain_adaptation()

    def _check_vars(self, scope, output_val=False):
        for v in tf.global_variables(scope=scope):
            print(v.name, "" if not output_val else self._sess.run(v))

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


def _test_classifier():
    da_type = global_defs.DA.A2R
    dl_src, dl_tgt = data_helper.read_paired_labeled_features(da_type)
    neurons = [[1024, 196, 128]]
    to_prt = []
    iterations=30000
    batch_sz = 256

    adda = ADDA(global_defs.mk_dir( os.path.join(global_defs.PATH_SAVING, 'ADDA_test')), [dl_src, dl_tgt])
    adda._train_classifier(iterations=2000)


def _test_discriminator():
    da_type = global_defs.DA.A2R
    dl_src, dl_tgt = data_helper.read_paired_labeled_features(da_type)
    neurons = [[1024, 196, 128]]
    to_prt = []
    iterations=30000
    batch_sz = 256
    adda = ADDA(global_defs.mk_dir( os.path.join(global_defs.PATH_SAVING, 'ADDA_test')), [dl_src, dl_tgt])
    adda._train_discriminator(iterations=1)


def _test_domain_adaptation():
    da_type = global_defs.DA.A2R
    dl_src, dl_tgt = data_helper.read_paired_labeled_features(da_type)
    neurons = [[1024, 196, 128]]
    to_prt = []
    iterations=30000
    batch_sz = 256
    adda = ADDA(global_defs.mk_dir( os.path.join(global_defs.PATH_SAVING, 'ADDA_test')), [dl_src, dl_tgt])
    #adda._domain_adaptation()
    #exit(0)
    adda._domain_adaptation()
    print('\n')
    #adda._domain_adaptation_tgt_encoder()


def _test_train():
    da_type = global_defs.DA.A2R
    dl_src, dl_tgt = data_helper.read_paired_labeled_features(da_type)
    neurons = [[1024, 196, 128]]
    to_prt = []
    iterations=30000
    batch_sz = 256
    adda = ADDA(global_defs.mk_dir( os.path.join(global_defs.PATH_SAVING, 'ADDA_test')), 
                [dl_src, dl_tgt], opt_step=0.0001, h_neurons_discriminator=[200])

    #adda.train(ADDA.Classifier, iterations=300000)
    #adda.train(ADDA.Discriminator, iterations=1)
    adda.train(ADDA.Discriminator | ADDA.DomainAdapt, iterations=10)
    #adda.train( ADDA.DomainAdapt, iterations=1)
    print('\n')
    #adda._domain_adaptation_tgt_encoder()


if __name__=='__main__':
    #_test_classifier()
    #_test_discriminator()
    #_test_domain_adaptation()
    _test_train()



    '''
src_encoder/l2_loss:0
src_encoder/Variable:0
src_encoder/w0:0
src_encoder/b0:0
src_encoder/Variable_1:0
src_encoder/w1:0
src_encoder/b1:0
tgt_encoder/l2_loss:0
tgt_encoder/Variable:0
tgt_encoder/w0:0
tgt_encoder/b0:0
tgt_encoder/Variable_1:0
tgt_encoder/w1:0
tgt_encoder/b1:0
tgt_encoder/w0/Adam:0
tgt_encoder/w0/Adam_1:0
tgt_encoder/b0/Adam:0
tgt_encoder/b0/Adam_1:0
tgt_encoder/w1/Adam:0
tgt_encoder/w1/Adam_1:0
tgt_encoder/b1/Adam:0
tgt_encoder/b1/Adam_1:0

DomainAdapt
src_encoder/l2_loss:0
src_encoder/Variable:0
src_encoder/w0:0
src_encoder/b0:0
src_encoder/Variable_1:0
src_encoder/w1:0
src_encoder/b1:0
src_encoder/l2_loss/Adam:0
src_encoder/l2_loss/Adam_1:0
tgt_encoder/l2_loss:0
tgt_encoder/Variable:0
tgt_encoder/w0:0
tgt_encoder/b0:0
tgt_encoder/Variable_1:0
tgt_encoder/w1:0
tgt_encoder/b1:0
tgt_encoder/l2_loss/Adam:0
tgt_encoder/l2_loss/Adam_1:0
    '''