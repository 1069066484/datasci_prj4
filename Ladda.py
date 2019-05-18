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
            #prev_dim = self._h_neurons_encoder[-1]
            prev_dim = self._label_count
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
                if i % 2000 == 0:
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
        #source_ft = self._src_encoder_yo
        self._construct_classifier(self._src_encoder_yo, self._src_encoder_l2_loss, reuse=False, trainable=False)
        source_ft = self._classifier_yo
        self._kp1 = self._classifier_keep_prob
        #target_ft = self._tgt_encoder_yo
        self._construct_classifier(self._tgt_encoder_yo, self._tgt_encoder_l2_loss, reuse=True, trainable=False)
        #src_encoder_acc = self._classifier_acc
        #src_encoder_feed_dict={self._classifier_keep_prob:1.0, self._src_encoder_keep_prob:1.0}
        target_ft = self._classifier_yo
        self._kp2 = self._classifier_keep_prob

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

        self._generator_loss = mapping_loss + self._tgt_encoder_l2_loss
        self._discriminator_loss = adversary_loss + self._discriminator_l2_loss


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
                    src_var_val = self._sess.run(src_v)
                    self._sess.run(tgt_v.assign(src_var_val))

    def _copy_classifier(self):
        clf_vars = tf.global_variables(scope=self._classifier_scope)
        len_half = round(len(clf_vars)/2)
        for i in range( len_half):
            var0 = self._sess.run(clf_vars[i+len_half])
            self._sess.run(clf_vars[i].assign(var0))

    def _init_uninited(self):
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                #print(var, ' caught')
                uninit_vars.append(var)
        self._sess.run(tf.variables_initializer(uninit_vars))

    def _train_discriminator(self, iterations=3000, batch_sz=64, do_clear=False):
        tf.reset_default_graph()
        self._construct_tgt_encoder(reuse=False, trainable=True)
        self._construct_src_encoder(reuse=False, trainable=False)
        self._build_ad_loss()

        with tf.Session() as self._sess:
            writer = tf.summary.FileWriter(os.path.join(self._saving_path,'discriminator/log'),self._sess.graph)

            saver_src_encoder = self._try_get_saver(self._src_encoder_scope, self._path_src_nn,  False, 
                                                    "Cannot find source encoder parameters", only_trainable=False)
            saver_tgt_encoder, loaded = self._try_get_saver(self._tgt_encoder_scope, self._path_tgt_nn, do_clear, ret_state=True, 
                                                            only_trainable=False)
            saver_discriminator = self._try_get_saver(self._discriminator_scope, self._path_dsc_nn, do_clear)

            saver_classifier = self._try_get_saver(self._classifier_scope, self._path_clf_nn, 
                                                   "Cannot find classifier parameters")

            if not loaded:
                #print("_copy_encoder")
                self._copy_encoder()

            tgt_encoder_train_variables = tf.trainable_variables(self._tgt_encoder_scope)
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=self._opt_step,beta1=0.5,
                                                    beta2=0.999).minimize(self._generator_loss,var_list=tgt_encoder_train_variables)
            disc_trainable_variables = tf.trainable_variables(self._discriminator_scope)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=self._opt_step,beta1=0.5,
                                                    beta2=0.999).minimize(self._discriminator_loss,var_list=disc_trainable_variables)
            self._init_uninited()
            #self._sess.run(tf.global_variables_initializer())

            for i in range(iterations):
                [src_dl, tgt_dl] = self._next_batch_discriminator(batch_sz)
                dict = {self._tgt_encoder_xi: tgt_dl[0], self._src_encoder_xi: src_dl[0], self._classifier_keep_prob:1.0,
                        self._tgt_encoder_keep_prob: self._train_keep_prob, self._discriminator_keep_prob: 1.0, self._kp2:1.0,self._kp1:1.0, self._src_encoder_keep_prob: 1.0, self._classifier_y:np.zeros([batch_sz , self._label_count])
                        } 
                optimizer_gen.run(feed_dict=dict)
                dict[self._tgt_encoder_keep_prob] = 1.0
                dict[self._discriminator_keep_prob] = self._train_keep_prob
                optimizer_disc.run(feed_dict=dict)
                if i % 400 == 0:
                    dict[self._discriminator_keep_prob] = 1.0

                    gen_loss = self._generator_loss.eval(feed_dict=dict)
                    disc_loss = self._discriminator_loss.eval(feed_dict=dict)

                    print("it:", i, 'gen_loss=',gen_loss,'disc_loss=',disc_loss,'target1=' #,tgt1, 'source1=',src1
                          )
                    self._histories_discriminator[0].append(gen_loss)
                    self._histories_discriminator[1].append(disc_loss)

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

            saver_src_encoder = self._try_get_saver(self._src_encoder_scope, self._path_src_nn,  False, 
                                                    "Cannot find source encoder parameters")
            saver_clf = self._try_get_saver(self._classifier_scope, self._path_clf_nn,  False, 
                                                    "Cannot find classifier parameters")
            saver_tgt_encoder = self._try_get_saver(self._tgt_encoder_scope, self._path_tgt_nn,  False, 
                                                    "Cannot find target encoder parameters")

            src_encoder_feed_dict.update({self._src_encoder_xi:self._src_test_dl[0], src_encoder_classifier_y:self._src_test_dl[1]})
            test_acc_src = src_encoder_acc.eval(feed_dict=src_encoder_feed_dict)
            src_data_encoded = self._src_encoder_yo.eval(feed_dict=src_encoder_feed_dict)

            src_encoder_feed_dict.update({self._src_encoder_xi:self._tgt_dl[0], src_encoder_classifier_y:self._tgt_dl[1]})
            test_acc_tgt = src_encoder_acc.eval(feed_dict=src_encoder_feed_dict)
            tgt_data_encoded = self._src_encoder_yo.eval(feed_dict=src_encoder_feed_dict)

            tgt_encoder_feed_dict.update({self._tgt_encoder_xi:self._tgt_dl[0], tgt_encoder_classifier_y:self._tgt_dl[1]})
            test_acc_tgt_adapted = tgt_encoder_acc.eval(feed_dict=tgt_encoder_feed_dict)
            tgt_data_encoded_adapted = self._tgt_encoder_yo.eval(feed_dict=tgt_encoder_feed_dict)
            print("Source domain test dataset acc:", test_acc_src, '\n',
                  "Target domain data set using source encoder acc:", test_acc_tgt, '\n',
                  "Target domain data set using target encoder acc:", test_acc_tgt_adapted, '\n')
            self._da_test_acc_src = test_acc_src
            self._da_test_acc_tgt = test_acc_tgt
            self._da_test_acc_tgt_adapted = test_acc_tgt_adapted
            #data_helper.visualize_da(src_data_encoded, tgt_data_encoded, tgt_data_encoded_adapted)

    def print_info(self):
        #self._history_classifier[0] train test loss
        i = self._history_classifier[1].index(max(self._history_classifier[1]))
        print('****************************************************************************\n',
            [self._input_dim] + self._h_neurons_encoder, self._h_neurons_classifier + [self._label_count], 
            self._h_neurons_discriminator +[2], self._opt_step, self._train_keep_prob, self._l2_reg,
            self._clf_batch_sz, self._disc_batch_sz,
            '\n'
              ,self._history_classifier[0][i], self._history_classifier[1][i], self._history_classifier[2][i],
              self._histories_discriminator[0][-1], self._histories_discriminator[1][-1], #gen disc loss
             self._da_test_acc_src, self._da_test_acc_tgt, self._da_test_acc_tgt_adapted, 
             '\n****************************************************************************')

    def train(self, train_net = Classifier | Discriminator | DomainAdapt,
              iterations = 20000, batch_sz=64, do_clear=False):
        if train_net & ADDA.Classifier != 0:
            self._clf_batch_sz = batch_sz
            print("\n\nClassifier Training Start")
            self._train_classifier(iterations, batch_sz, do_clear)
        if train_net & ADDA.Discriminator != 0:
            self._disc_batch_sz = batch_sz
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


def main():
    da_type = global_defs.DA.A2R
    dl_src, dl_tgt = data_helper.read_paired_labeled_features(da_type)
    to_prt = []
    iterations=30000
    batch_sz = 256

    prt = []


    for h_neurons_discriminator in ([[i] for i in list(range(10,200,30))] + 
                                    ADDA.neuron_arrange(list(range(10,200,35)),2)):
        print("\n\nh_neurons_discriminator=",h_neurons_discriminator)
        adda = ADDA(global_defs.mk_dir( os.path.join(global_defs.PATH_SAVING, 'ADDA_test')), 
                    [dl_src, dl_tgt], opt_step=0.00005, h_neurons_discriminator=h_neurons_discriminator, h_neurons_encoder=[1024, 196],
                    h_neurons_classifier=[128])
        #1024, 196, 128
        #adda.train(ADDA.Classifier, iterations=20000, batch_sz=batch_sz)

        adda.train(ADDA.Discriminator | ADDA.DomainAdapt, iterations=250,  batch_sz=batch_sz, do_clear=True)
        prt.append([h_neurons_discriminator, adda._da_test_acc_src, adda._da_test_acc_tgt, adda._da_test_acc_tgt_adapted])
    #adda.print_info()
    for p in prt:
        print(p,'\n')

    print('\n')


def main2():
    da_type = global_defs.DA.A2R
    dl_src, dl_tgt = data_helper.read_paired_labeled_features(da_type)
    to_prt = []
    iterations=30000
    batch_sz = 256
    h_neurons_discriminator = []
    prt = []
    print("\n\nh_neurons_discriminator=",h_neurons_discriminator)
    adda = ADDA(global_defs.mk_dir( os.path.join(global_defs.PATH_SAVING, 'ADDA_test')), 
                [dl_src, dl_tgt], opt_step=0.000005, h_neurons_discriminator=h_neurons_discriminator, h_neurons_encoder=[1024, 196],
                h_neurons_classifier=[128],l2=0.0)
    #1024, 196, 128
    #adda.train(ADDA.Classifier, iterations=20000, batch_sz=batch_sz)

    adda.train(ADDA.Discriminator | ADDA.DomainAdapt, iterations=100,  batch_sz=batch_sz, do_clear=True)
    prt.append([h_neurons_discriminator, adda._da_test_acc_src, adda._da_test_acc_tgt, adda._da_test_acc_tgt_adapted])
    #adda.print_info()
    for p in prt:
        print(p,'\n')

    print('\n')


if __name__=='__main__':
    #_test_classifier()
    #_test_discriminator()
    #_test_domain_adaptation()
    #main()
    main2()



    '''
Source domain test dataset acc: 0.81237113
 Target domain data set using source encoder acc: 0.71287584
 Target domain data set using target encoder acc: 0.019508837

****************************************************************************
 [2048, 1024, 196] [128, 65] [128, 2] 0.0001 0.5 0.005 256 256
 0.99609375 0.81237113 0.18289715 1.3103122 0.3143495 0.81237113 0.71287584 0.019508837
****************************************************************************

[[10], 0.92268044, 0.71287584, 0.6981868]

[[15], 0.92268044, 0.71287584, 0.6915309]

[[20], 0.92268044, 0.71287584, 0.7092036]

[[25], 0.92268044, 0.71287584, 0.7165481]

[[30], 0.92268044, 0.71287584, 0.7050723]

[[35], 0.92268044, 0.71287584, 0.70989215]

[[40], 0.92268044, 0.71287584, 0.70231813]

[[45], 0.92268044, 0.71287584, 0.7036952]

[[50], 0.92268044, 0.71287584, 0.71195775]

[[55], 0.92268044, 0.71287584, 0.7092036]

[[60], 0.92268044, 0.71287584, 0.6958917]

[[65], 0.92268044, 0.71287584, 0.6952031]

[[70], 0.92268044, 0.71287584, 0.7082855]

[[75], 0.92268044, 0.71287584, 0.7094331]

[[80], 0.92268044, 0.71287584, 0.7073675]

[[85], 0.92268044, 0.71287584, 0.69749826]

[[90], 0.92268044, 0.71287584, 0.7092036]

[[95], 0.92268044, 0.71287584, 0.71012163]

[[100], 0.92268044, 0.71287584, 0.6935965]

[[105], 0.92268044, 0.71287584, 0.700482]

[[110], 0.92268044, 0.71287584, 0.700941]

[[115], 0.92268044, 0.71287584, 0.7016296]

[[120], 0.92268044, 0.71287584, 0.70140004]

[[125], 0.92268044, 0.71287584, 0.7034657]

[[130], 0.92268044, 0.71287584, 0.70713794]

[[135], 0.92268044, 0.71287584, 0.7082855]

[[140], 0.92268044, 0.71287584, 0.7066789]

[[145], 0.92268044, 0.71287584, 0.7059904]

[[150], 0.92268044, 0.71287584, 0.700482]

[[155], 0.92268044, 0.71287584, 0.70759696]

[[160], 0.92268044, 0.71287584, 0.7053018]

[[165], 0.92268044, 0.71287584, 0.69405556]

[[170], 0.92268044, 0.71287584, 0.7082855]

[[175], 0.92268044, 0.71287584, 0.7073675]

[[180], 0.92268044, 0.71287584, 0.70989215]

[[185], 0.92268044, 0.71287584, 0.7039247]

[[190], 0.92268044, 0.71287584, 0.7087445]

[[195], 0.92268044, 0.71287584, 0.70897406]

[(190, 190), 0.92268044, 0.71287584, 0.69658023]

[(190, 180), 0.92268044, 0.71287584, 0.7007115]

[(190, 170), 0.92268044, 0.71287584, 0.7092036]

[(190, 160), 0.92268044, 0.71287584, 0.7117283]

[(190, 150), 0.92268044, 0.71287584, 0.7183842]

[(190, 140), 0.92268044, 0.71287584, 0.70185906]

[(190, 130), 0.92268044, 0.71287584, 0.7073675]

[(190, 120), 0.92268044, 0.71287584, 0.7025476]

[(190, 110), 0.92268044, 0.71287584, 0.70851505]

[(190, 100), 0.92268044, 0.71287584, 0.7050723]

[(190, 90), 0.92268044, 0.71287584, 0.7131053]

[(190, 80), 0.92268044, 0.71287584, 0.7011705]

[(190, 70), 0.92268044, 0.71287584, 0.7020886]

[(190, 60), 0.92268044, 0.71287584, 0.70713794]

[(190, 50), 0.92268044, 0.71287584, 0.69566214]

[(190, 40), 0.92268044, 0.71287584, 0.69658023]

[(190, 30), 0.92268044, 0.71287584, 0.71287584]

[(190, 20), 0.92268044, 0.71287584, 0.7064494]

[(190, 10), 0.92268044, 0.71287584, 0.71333486]

[(180, 180), 0.92268044, 0.71287584, 0.7036952]

[(180, 170), 0.92268044, 0.71287584, 0.70576084]

[(180, 160), 0.92268044, 0.71287584, 0.70323616]

[(180, 150), 0.92268044, 0.71287584, 0.7020886]

[(180, 140), 0.92268044, 0.71287584, 0.7039247]

[(180, 130), 0.92268044, 0.71287584, 0.7011705]

[(180, 120), 0.92268044, 0.71287584, 0.7034657]

[(180, 110), 0.92268044, 0.71287584, 0.70415425]

[(180, 100), 0.92268044, 0.71287584, 0.7055313]

[(180, 90), 0.92268044, 0.71287584, 0.700482]

[(180, 80), 0.92268044, 0.71287584, 0.70851505]

[(180, 70), 0.92268044, 0.71287584, 0.7011705]

[(180, 60), 0.92268044, 0.71287584, 0.7165481]

[(180, 50), 0.92268044, 0.71287584, 0.7117283]

[(180, 40), 0.92268044, 0.71287584, 0.7110397]

[(180, 30), 0.92268044, 0.71287584, 0.7064494]

[(180, 20), 0.92268044, 0.71287584, 0.7011705]

[(180, 10), 0.92268044, 0.71287584, 0.7059904]

[(170, 170), 0.92268044, 0.71287584, 0.7020886]

[(170, 160), 0.92268044, 0.71287584, 0.70989215]

[(170, 150), 0.92268044, 0.71287584, 0.70185906]

[(170, 140), 0.92268044, 0.71287584, 0.7016296]

[(170, 130), 0.92268044, 0.71287584, 0.69749826]

[(170, 120), 0.92268044, 0.71287584, 0.71012163]

[(170, 110), 0.92268044, 0.71287584, 0.7174662]

[(170, 100), 0.92268044, 0.71287584, 0.7011705]

[(170, 90), 0.92268044, 0.71287584, 0.6945146]

[(170, 80), 0.92268044, 0.71287584, 0.70851505]

[(170, 70), 0.92268044, 0.71287584, 0.71563]

[(170, 60), 0.92268044, 0.71287584, 0.708056]

[(170, 50), 0.92268044, 0.71287584, 0.7094331]

[(170, 40), 0.92268044, 0.71287584, 0.7011705]

[(170, 30), 0.92268044, 0.71287584, 0.71333486]

[(170, 20), 0.92268044, 0.71287584, 0.7087445]

[(170, 10), 0.92268044, 0.71287584, 0.7179252]

[(160, 160), 0.92268044, 0.71287584, 0.7078265]

[(160, 150), 0.92268044, 0.71287584, 0.7069084]

[(160, 140), 0.92268044, 0.71287584, 0.70323616]

[(160, 130), 0.92268044, 0.71287584, 0.7053018]

[(160, 120), 0.92268044, 0.71287584, 0.7055313]

[(160, 110), 0.92268044, 0.71287584, 0.70713794]

[(160, 100), 0.92268044, 0.71287584, 0.7195318]

[(160, 90), 0.92268044, 0.71287584, 0.7078265]

[(160, 80), 0.92268044, 0.71287584, 0.7039247]

[(160, 70), 0.92268044, 0.71287584, 0.7108102]

[(160, 60), 0.92268044, 0.71287584, 0.70438373]

[(160, 50), 0.92268044, 0.71287584, 0.71035117]

[(160, 40), 0.92268044, 0.71287584, 0.7144824]

[(160, 30), 0.92268044, 0.71287584, 0.7131053]

[(160, 20), 0.92268044, 0.71287584, 0.69887537]

[(160, 10), 0.92268044, 0.71287584, 0.7126463]

[(150, 150), 0.92268044, 0.71287584, 0.7108102]

[(150, 140), 0.92268044, 0.71287584, 0.7059904]

[(150, 130), 0.92268044, 0.71287584, 0.70759696]

[(150, 120), 0.92268044, 0.71287584, 0.7025476]

[(150, 110), 0.92268044, 0.71287584, 0.7078265]

[(150, 100), 0.92268044, 0.71287584, 0.71287584]

[(150, 90), 0.92268044, 0.71287584, 0.7087445]

[(150, 80), 0.92268044, 0.71287584, 0.708056]

[(150, 70), 0.92268044, 0.71287584, 0.70759696]

[(150, 60), 0.92268044, 0.71287584, 0.7082855]

[(150, 50), 0.92268044, 0.71287584, 0.71012163]

[(150, 40), 0.92268044, 0.71287584, 0.7131053]

[(150, 30), 0.92268044, 0.71287584, 0.6952031]

[(150, 20), 0.92268044, 0.71287584, 0.70277715]

[(150, 10), 0.92268044, 0.71287584, 0.70185906]

[(140, 140), 0.92268044, 0.71287584, 0.7092036]

[(140, 130), 0.92268044, 0.71287584, 0.7144824]

[(140, 120), 0.92268044, 0.71287584, 0.7108102]

[(140, 110), 0.92268044, 0.71287584, 0.7078265]

[(140, 100), 0.92268044, 0.71287584, 0.71195775]

[(140, 90), 0.92268044, 0.71287584, 0.700482]

[(140, 80), 0.92268044, 0.71287584, 0.70461327]

[(140, 70), 0.92268044, 0.71287584, 0.7048428]

[(140, 60), 0.92268044, 0.71287584, 0.71333486]

[(140, 50), 0.92268044, 0.71287584, 0.7030067]

[(140, 40), 0.92268044, 0.71287584, 0.7108102]

[(140, 30), 0.92268044, 0.71287584, 0.7020886]

[(140, 20), 0.92268044, 0.71287584, 0.7149415]

[(140, 10), 0.92268044, 0.71287584, 0.6977278]

[(130, 130), 0.92268044, 0.71287584, 0.69703925]

[(130, 120), 0.92268044, 0.71287584, 0.70576084]

[(130, 110), 0.92268044, 0.71287584, 0.7073675]

[(130, 100), 0.92268044, 0.71287584, 0.71608907]

[(130, 90), 0.92268044, 0.71287584, 0.7131053]

[(130, 80), 0.92268044, 0.71287584, 0.7073675]

[(130, 70), 0.92268044, 0.71287584, 0.7110397]

[(130, 60), 0.92268044, 0.71287584, 0.71287584]

[(130, 50), 0.92268044, 0.71287584, 0.7087445]

[(130, 40), 0.92268044, 0.71287584, 0.7144824]

[(130, 30), 0.92268044, 0.71287584, 0.7121873]

[(130, 20), 0.92268044, 0.71287584, 0.70621985]

[(130, 10), 0.92268044, 0.71287584, 0.7036952]

[(120, 120), 0.92268044, 0.71287584, 0.7117283]

[(120, 110), 0.92268044, 0.71287584, 0.700941]

[(120, 100), 0.92268044, 0.71287584, 0.71195775]

[(120, 90), 0.92268044, 0.71287584, 0.70576084]

[(120, 80), 0.92268044, 0.71287584, 0.70002294]

[(120, 70), 0.92268044, 0.71287584, 0.7094331]

[(120, 60), 0.92268044, 0.71287584, 0.71333486]

[(120, 50), 0.92268044, 0.71287584, 0.70461327]

[(120, 40), 0.92268044, 0.71287584, 0.7094331]

[(120, 30), 0.92268044, 0.71287584, 0.7066789]

[(120, 20), 0.92268044, 0.71287584, 0.7066789]

[(120, 10), 0.92268044, 0.71287584, 0.71563]

[(110, 110), 0.92268044, 0.71287584, 0.708056]

[(110, 100), 0.92268044, 0.71287584, 0.71149874]

[(110, 90), 0.92268044, 0.71287584, 0.70989215]

[(110, 80), 0.92268044, 0.71287584, 0.71287584]

[(110, 70), 0.92268044, 0.71287584, 0.69658023]

[(110, 60), 0.92268044, 0.71287584, 0.71195775]

[(110, 50), 0.92268044, 0.71287584, 0.70277715]

[(110, 40), 0.92268044, 0.71287584, 0.6993344]

[(110, 30), 0.92268044, 0.71287584, 0.7197613]

[(110, 20), 0.92268044, 0.71287584, 0.7066789]

[(110, 10), 0.92268044, 0.71287584, 0.700482]

[(100, 100), 0.92268044, 0.71287584, 0.71287584]

[(100, 90), 0.92268044, 0.71287584, 0.71425295]

[(100, 80), 0.92268044, 0.71287584, 0.70461327]

[(100, 70), 0.92268044, 0.71287584, 0.7094331]

[(100, 60), 0.92268044, 0.71287584, 0.70185906]

[(100, 50), 0.92268044, 0.71287584, 0.7149415]

[(100, 40), 0.92268044, 0.71287584, 0.69864583]

[(100, 30), 0.92268044, 0.71287584, 0.7108102]

[(100, 20), 0.92268044, 0.71287584, 0.6947441]

[(100, 10), 0.92268044, 0.71287584, 0.71035117]

[(90, 90), 0.92268044, 0.71287584, 0.7121873]

[(90, 80), 0.92268044, 0.71287584, 0.69887537]

[(90, 70), 0.92268044, 0.71287584, 0.7016296]

[(90, 60), 0.92268044, 0.71287584, 0.7096626]

[(90, 50), 0.92268044, 0.71287584, 0.7020886]

[(90, 40), 0.92268044, 0.71287584, 0.70415425]

[(90, 30), 0.92268044, 0.71287584, 0.7048428]

[(90, 20), 0.92268044, 0.71287584, 0.70621985]

[(90, 10), 0.92268044, 0.71287584, 0.7117283]

[(80, 80), 0.92268044, 0.71287584, 0.7096626]

[(80, 70), 0.92268044, 0.71287584, 0.7078265]

[(80, 60), 0.92268044, 0.71287584, 0.6958917]

[(80, 50), 0.92268044, 0.71287584, 0.7154005]

[(80, 40), 0.92268044, 0.71287584, 0.7064494]

[(80, 30), 0.92268044, 0.71287584, 0.7124168]

[(80, 20), 0.92268044, 0.71287584, 0.70277715]

[(80, 10), 0.92268044, 0.71287584, 0.6981868]

[(70, 70), 0.92268044, 0.71287584, 0.7053018]

[(70, 60), 0.92268044, 0.71287584, 0.7069084]

[(70, 50), 0.92268044, 0.71287584, 0.69795734]

[(70, 40), 0.92268044, 0.71287584, 0.7092036]

[(70, 30), 0.92268044, 0.71287584, 0.7066789]

[(70, 20), 0.92268044, 0.71287584, 0.7087445]

[(70, 10), 0.92268044, 0.71287584, 0.7025476]

[(60, 60), 0.92268044, 0.71287584, 0.69887537]

[(60, 50), 0.92268044, 0.71287584, 0.7069084]

[(60, 40), 0.92268044, 0.71287584, 0.70140004]

[(60, 30), 0.92268044, 0.71287584, 0.71058065]

[(60, 20), 0.92268044, 0.71287584, 0.7016296]

[(60, 10), 0.92268044, 0.71287584, 0.71035117]

[(50, 50), 0.92268044, 0.71287584, 0.70713794]

[(50, 40), 0.92268044, 0.71287584, 0.7144824]

[(50, 30), 0.92268044, 0.71287584, 0.7066789]

[(50, 20), 0.92268044, 0.71287584, 0.692908]

[(50, 10), 0.92268044, 0.71287584, 0.70415425]

[(40, 40), 0.92268044, 0.71287584, 0.70140004]

[(40, 30), 0.92268044, 0.71287584, 0.69313747]

[(40, 20), 0.92268044, 0.71287584, 0.7073675]

[(40, 10), 0.92268044, 0.71287584, 0.7073675]

[(30, 30), 0.92268044, 0.71287584, 0.7087445]

[(30, 20), 0.92268044, 0.71287584, 0.70897406]

[(30, 10), 0.92268044, 0.71287584, 0.71035117]

[(20, 20), 0.92268044, 0.71287584, 0.7131053]

[(20, 10), 0.92268044, 0.71287584, 0.7117283]

[(10, 10), 0.92268044, 0.71287584, 0.70621985]
    '''