# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: Implementation of a general classifier using neural network.
"""
import numpy as np
import Ldata_helper as data_helper
import tensorflow as tf
import Lglobal_defs as global_defs
import os
import itertools


class GeneralNNClassifier:
    def __init__(self, saving_path, labeled_data=None, h_neurons=[4096, 1024, 256], 
                 train_test_split=0.6, train_dl=None, test_dl=None, opt_step=0.02, l2=0.5,
                 keep_prob=0.5):
        self._saving_path = saving_path
        if labeled_data is not None:
            #print(labeled_data)
            labeled_data = list(data_helper.shuffle_labeled_data(labeled_data))
            labeled_data[1] = data_helper.labels2one_hot(labeled_data[1])
            [train_dl, test_dl] = data_helper.labeled_data_split(labeled_data, train_test_split)
        else:
            self._data, self._labels = train_dl
            test_dl = test_dl
        self._t_ld = test_dl
        self._labels = train_dl[1]
        self._data = train_dl[0]
        self._input_dim = self._data.shape[1]
        self._label_count = self._labels.shape[1]
        self._curr_batch_idx = 0
        self._opt_step = opt_step
        self._l2_loss = None
        self._l2_reg = l2
        self._h_neurons = h_neurons
        self.w_generator = GeneralNNClassifier._w_generator
        self.b_generator = GeneralNNClassifier._b_generator
        self._train_keep_prob = keep_prob
        self._histories = [[],[],[]] #training, test, loss

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

    def _add_layer2(self, input, in_sz, out_sz, act_fun=None):
        #w = tf.Variable(tf.truncated_normal([in_sz, out_sz],stddev=0.1))
        w = tf.Variable(self.w_generator(w_shape=[in_sz,out_sz]))
        self._l2_loss += tf.nn.l2_loss(w) * self._l2_reg
        #tf.assign_add(self._l2_loss, tf.nn.l2_loss(w) * self._l2_reg)
        b = tf.Variable(self.b_generator(b_shape=[out_sz]))

        wx_plusb = tf.matmul(input, w) + b
        output = wx_plusb if act_fun is None else act_fun(wx_plusb)
        return tf.nn.dropout(output, self._keep_prob)

    def _add_layer(self, input, in_sz, out_sz, act_fun=None):
        output = tf.layers.dense(input,out_sz,activation=act_fun,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg),
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.15),
                                 bias_initializer=tf.random_uniform_initializer(-0.2,0.2)
                                 ) 
        return tf.nn.dropout(output, self._keep_prob)

    def _construct_nn(self):
        self._l2_loss = tf.Variable(tf.constant(0.0))
        add_layer = self._add_layer
        self._keep_prob = tf.placeholder(tf.float32)
        self._x = tf.placeholder(tf.float32, [None, self._input_dim])
        self._y = tf.placeholder(tf.float32, [None, self._label_count])

        h = self._x
        prev_dim = self._input_dim
        for neurons in self._h_neurons:
            h = add_layer(h, prev_dim, neurons, act_fun=tf.nn.relu)
            prev_dim = neurons
        #h1 = add_layer(self._x, self._input_dim, 64, act_fun=tf.nn.relu)
        #h2 = add_layer(h1, 64, 32, act_fun=tf.nn.relu)
        self._yo = add_layer(h, prev_dim, self._label_count, act_fun=None)
        
        #loss = -tf.reduce_sum(self._y*tf.log(self._yo))
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._yo)) + self._l2_loss
        #self._loss = tf.reduce_sum(tf.square(self._y - self._yo))  + self._l2_loss
        self._trainer = tf.train.AdamOptimizer(self._opt_step).minimize(self._loss)
        #self._trainer = tf.train.GradientDescentOptimizer(self._opt_step*500).minimize(self._loss)

        correct_prediction = tf.equal(tf.argmax(self._yo, 1), tf.argmax(self._y, 1))
        self._acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _next_batch(self, batch_sz):
        indices = list(range(self._curr_batch_idx, self._curr_batch_idx+batch_sz))
        self._curr_batch_idx = (batch_sz + self._curr_batch_idx) % self._data.shape[0]
        indices = [i-self._data.shape[0] if i >= self._data.shape[0] else i for i in indices]
        return [self._data[indices], self._labels[indices]]
    
    def _eval(self, labeled_data):
        return self._acc.eval(feed_dict={self._x: labeled_data[0], self._y:labeled_data[1], self._keep_prob:1.0})

    def history(self):
        return self._histories

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
            if i % 1000 == 999:
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


def read_mnist():
    import pickle
    p = os.path.join(global_defs.PATH_SAVING, "mnist_NNClassifier_test")
    p = data_helper.pkfilename(p)
    if os.path.exists(p):
        return pickle.load(open(p,'rb'))
    mnist = data_helper.read_mnist(one_hot=True)
    pickle.dump(mnist, open(p ,'wb'))
    return mnist


def _test():
    batchsize=64
    mnist = data_helper.read_mnist(one_hot=True)
    def train(init_bias=-0.1): 

        sess = tf.InteractiveSession()
        #---------------------------------------初始化网络结构-------------------------------------
        x = tf.placeholder("float", [None, 784],name='x-input')
        y_ = tf.placeholder("float", [None,10],name='y-input')
        
        W1 = tf.Variable(tf.random_uniform([784,100],-0.5+init_bias,0.5+init_bias))
        b1 = tf.Variable(tf.random_uniform([100],-0.5+init_bias,0.5+init_bias))
        u1 = tf.matmul(x,W1) + b1
        y1 = tf.nn.sigmoid(u1)
    #    y1=u1
        W2 = tf.Variable(tf.random_uniform([100,10],-0.5+init_bias,0.5+init_bias))
        b2 = tf.Variable(tf.random_uniform([10],-0.5+init_bias,0.5+init_bias))
        y = tf.nn.sigmoid(tf.matmul(y1,W2) + b2)
        #---------------------------------------设置网络的训练方式-------------------------------------
        mse = tf.reduce_sum(tf.square(y-y_))#mse
    #    train_step = tf.train.GradientDescentOptimizer(0.02).minimize(mse)
        train_step = tf.train.AdamOptimizer(0.001).minimize(mse)
    
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
        init = tf.global_variables_initializer()
        sess.run(init)
        #---------------------------------------开始训练-------------------------------------
        for i in range(1001):
          batch_xs, batch_ys = mnist.train.next_batch(batchsize)
          sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) 

        print ('权重初始化范围[%.1f,%.1f],1000次训练过后的准确率'
               %(init_bias-0.5,init_bias+0.5),sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    train()


def _main_mnist_test():
    mnist = read_mnist()
    print('\n\n\n\n\n',mnist.test.images.shape, mnist.test.labels.shape) #(10000, 784) (10000, 10)
    print('\n\n\n\n\n',mnist.train.images.shape, mnist.train.labels.shape)#(55000, 784) (55000, 10)
    neurons = [32, 128, 512]
    to_prt = []
    for l2 in [0.0]:
        for opt_step in [0.001]:
            for h_neurons in GeneralNNClassifier.neuron_arrange(neurons, 3):
                to_prt.append([l2, opt_step])
                print('\n\n',l2, opt_step, h_neurons)
                classifier = GeneralNNClassifier(os.path.join(global_defs.PATH_SAVING, 'NN_classifier_test'), train_dl=[mnist.train.images, mnist.train.labels], test_dl=[mnist.test.images, mnist.test.labels], opt_step=opt_step, l2=l2, h_neurons=h_neurons)
                classifier.train(batch_sz=64, iterations=3000)
                hist = classifier.history()
                to_prt[-1] += [classifier.neurons(), hist[0][-1], max(hist[1]), hist[2][-1]]
    print("\n\n")
    for p in to_prt:
        print(p)


def _main_DA_test():
    da_type = global_defs.DA.A2R
    dl_src, _ = data_helper.read_paired_labeled_features(da_type)
    #64 labels
    #print(dl_src)
    neurons = [96, 196, 384]
    to_prt = []
    keep_prob = 0.7
    for l2 in [0.0]:
        for opt_step in [0.005]:
            for h_neurons in GeneralNNClassifier.neuron_arrange(neurons, 3):
                to_prt.append([l2, opt_step])
                print('\n\n',l2, opt_step, h_neurons)
                classifier = GeneralNNClassifier(os.path.join(global_defs.PATH_SAVING, 'NN_classifier_test'), labeled_data=dl_src, opt_step=opt_step, l2=l2, h_neurons=h_neurons, keep_prob=keep_prob)
                classifier.train(batch_sz=64, iterations=3000)
                hist = classifier.history()
                to_prt[-1] += [classifier.neurons(), hist[0][-1], hist[1][-1], hist[2][-1], classifier._train_keep_prob]
    print("\n\n")
    print(da_type, " pos_neurons=",neurons)
    for p in to_prt:
        print(p)


def _main_DA_test_detail():
    da_type = global_defs.DA.A2R
    dl_src, _ = data_helper.read_paired_labeled_features(da_type)
    #64 labels
    #print(dl_src)
    neurons = GeneralNNClassifier.neuron_arrange([128, 196, 256, 512],2, True)
    to_prt = []
    iterations=15000
    batch_sz = 128
    for l2 in [0.000015]:
        for opt_step in [0.001]:
            #for h_neurons in GeneralNNClassifier.neuron_arrange(neurons, 3):
            for h_neurons in neurons:
                for keep_prob in [0.3,0.5,0.7]:
                    to_prt.append([l2, opt_step])
                    
                    classifier = GeneralNNClassifier(os.path.join(global_defs.PATH_SAVING, 'NN_classifier_test'), labeled_data=dl_src, opt_step=opt_step, l2=l2, h_neurons=h_neurons, keep_prob=keep_prob)
                    classifier.train(batch_sz=batch_sz, iterations=iterations)
                    hist = classifier.history()
                    to_prt[-1] += [classifier.neurons(), hist[0][-1], hist[1][-1], hist[2][-1], classifier._train_keep_prob, batch_sz, iterations]
                    print(to_prt[-1],'  \n')
    print("\n\n")
    print(da_type, " pos_neurons=",neurons)
    for p in to_prt:
        print(p)


def _main_DA_test_detail3():
    da_type = global_defs.DA.A2R
    dl_src, _ = data_helper.read_paired_labeled_features(da_type)
    #64 labels
    #print(dl_src)
    #neurons = GeneralNNClassifier.neuron_arrange([128, 196, 256, 512, 1024], 3, True)
    neurons = [[256]]
    #neurons = [[384, 196, 96],[1024, 196, 128],[512,128],[256]]
    #neurons = [[300],[256],[230]]
    #[0.0009, 0.00015, [2048, 1024, 196, 128, 65], 1.0, 0.81237113, -4.318195, 0.5, 256, 30000]
    to_prt = []
    iterations=5000
    batch_sz = 256
    for batch_sz in [256]:
        for l2 in [0.00001,100.0]:
            for opt_step in [0.005]:
                #for h_neurons in GeneralNNClassifier.neuron_arrange(neurons, 3):
                for h_neurons in neurons:
                    for keep_prob in [0.5]:
                        to_prt.append([l2, opt_step])
                    
                        classifier = GeneralNNClassifier(os.path.join(global_defs.PATH_SAVING, 'NN_classifier_test'), labeled_data=dl_src, opt_step=opt_step, l2=l2, h_neurons=h_neurons, keep_prob=keep_prob)
                        classifier.train(batch_sz=batch_sz, iterations=iterations)
                        hist = classifier.history()
                        to_prt[-1] += [classifier.neurons(), hist[0][-1], max(hist[1]), hist[2][-1], classifier._train_keep_prob, batch_sz, iterations]
                        print(to_prt[-1],'  \n')
    print("\n\n")
    print(da_type, " pos_neurons=",neurons)
    for p in to_prt:
        print(p)


def _test_neuron_arrange():
    neurons = [32, 64, 128, 256, 512, 1024]
    print(GeneralNNClassifier.neuron_arrange(neurons, 3))


'''
Greater l2 allows greater opt_step
'''
if __name__=='__main__':
    #_test_neuron_arrange()
    #_main_mnist_test()
    #_test()
    #_main_DA_test()
    #_main_DA_test_detail()
    _main_DA_test_detail3()

"""
DA.A2R  pos_neurons= [[1024, 196, 128]]
[7e-06, 0.0005, [2048, 1024, 196, 128, 65], 1.0, 0.8030928, -9.033175, 0.9, 128, 25000] sigmoid config

DA.A2R  pos_neurons= [[1024, 196, 128]]
[0.0007, 0.00015, [2048, 1024, 196, 128, 65], 0.99609375, 0.8103093, -3.5037632, 0.5, 256, 25000] relu config

DA.A2R  pos_neurons= [[1024, 196, 128]]
[0.0015, 0.00015, [2048, 1024, 196, 128, 65], 1.0, 0.8051546, -3.4865575, 0.5, 256, 25000] relu config

DA.A2R  pos_neurons= [[1024, 196, 128]]
[0.001, 0.00015, [2048, 1024, 196, 128, 65], 1.0, 0.7773196, -3.5664134, 0.5, 512, 25000] relu config

[0.0009, 0.00015, [2048, 1024, 196, 128, 65], 1.0, 0.80206186, -4.147231, 0.5, 32, 30000]
[0.0009, 0.00015, [2048, 1024, 196, 128, 65], 1.0, 0.785567, -4.2574863, 0.5, 64, 30000]
[0.0009, 0.00015, [2048, 1024, 196, 128, 65], 0.9921875, 0.80721647, -4.2831287, 0.5, 128, 30000]
[0.0009, 0.00015, [2048, 1024, 196, 128, 65], 1.0, 0.81237113, -4.318195, 0.5, 256, 30000]
[0.0009, 0.00015, [2048, 1024, 196, 128, 65], 1.0, 0.79690725, -4.3374057, 0.5, 512, 30000]
[0.0009, 0.00015, [2048, 1024, 196, 128, 65], 1.0, 0.80618554, -4.347394, 0.5, 1024, 30000]

DA.A2R  pos_neurons= [[1024], [512], [256], [128]]
[0.0009, 5e-05, [2048, 1024, 65], 0.9921875, 0.80103093, -2.3772213, 0.5, 256, 50000]
[0.0009, 5e-05, [2048, 512, 65], 1.0, 0.79072165, -2.3843088, 0.5, 256, 50000]
[0.0009, 5e-05, [2048, 256, 65], 1.0, 0.8340206, -2.3719723, 0.5, 256, 50000]
[0.0009, 5e-05, [2048, 128, 65], 1.0, 0.78659797, -2.3555489, 0.5, 256, 50000]

[0.005, 5e-05, [2048, 256, 65], 0.99609375, 0.83092785, -1.9657521, 0.5, 256, 50000]

DA.A2R  pos_neurons= [[300], [256], [230]]
[0.0009, 5e-05, [2048, 300, 65], 0.99609375, 0.80206186, -2.370272, 0.5, 256, 50000]
[0.0009, 5e-05, [2048, 256, 65], 0.99609375, 0.7886598, -2.3727667, 0.5, 256, 50000]
[0.0009, 5e-05, [2048, 230, 65], 1.0, 0.8, -2.3794425, 0.5, 256, 50000]
[0.005, 5e-05, [2048, 300, 65], 1.0, 0.8, -1.9751942, 0.5, 256, 50000]
[0.005, 5e-05, [2048, 256, 65], 0.9921875, 0.80206186, -1.9685702, 0.5, 256, 50000]
[0.005, 5e-05, [2048, 230, 65], 0.99609375, 0.80206186, -1.9743768, 0.5, 256, 50000]
[0.003, 5e-05, [2048, 300, 65], 0.99609375, 0.8154639, -2.1508727, 0.5, 256, 50000]
[0.003, 5e-05, [2048, 256, 65], 0.99609375, 0.814433, -2.1471484, 0.5, 256, 50000]
[0.003, 5e-05, [2048, 230, 65], 0.99609375, 0.81340206, -2.1636033, 0.5, 256, 50000]
[0.01, 5e-05, [2048, 300, 65], 1.0, 0.79484534, -1.6272887, 0.5, 256, 50000]
[0.01, 5e-05, [2048, 256, 65], 1.0, 0.80721647, -1.6222845, 0.5, 256, 50000]
[0.01, 5e-05, [2048, 230, 65], 1.0, 0.80618554, -1.644438, 0.5, 256, 50000]

DA.A2R  pos_neurons= [[1024, 256, 128]]
[0.0009, 5e-05, [2048, 1024, 256, 128, 65], 1.0, 0.79484534, -3.8444664, 0.5, 256, 80000]
[0.005, 5e-05, [2048, 1024, 256, 128, 65], 1.0, 0.7824742, -3.4252918, 0.5, 256, 80000]
[0.003, 5e-05, [2048, 1024, 256, 128, 65], 0.99609375, 0.7938144, -3.6174607, 0.5, 256, 80000]
[0.01, 5e-05, [2048, 1024, 256, 128, 65], 1.0, 0.7639175, -2.956146, 0.5, 256, 80000]
[0.03, 5e-05, [2048, 1024, 256, 128, 65], 0.88671875, 0.6969072, -1.6625171, 0.5, 256, 80000]

[0.0009, 5e-05, [2048, 256, 128, 65], 0.99609375, 0.80618554, -3.8362322, 0.5, 256, 80000]
[0.005, 5e-05, [2048, 256, 128, 65], 1.0, 0.79793817, -3.3890295, 0.5, 256, 80000]
[0.003, 5e-05, [2048, 256, 128, 65], 0.99609375, 0.80206186, -3.6005623, 0.5, 256, 80000]
[0.01, 5e-05, [2048, 256, 128, 65], 0.99609375, 0.76701033, -2.974491, 0.5, 256, 80000]
[0.03, 5e-05, [2048, 256, 128, 65], 0.88671875, 0.72164947, -1.7221055, 0.5, 256, 80000]

DA.A2R  pos_neurons= [[512, 128]]
[0.0009, 5e-05, [2048, 512, 128, 65], 1.0, 0.785567, -4.861924, 0.5, 256, 100000]
[0.005, 5e-05, [2048, 512, 128, 65], 0.99609375, 0.7876289, -4.4389668, 0.5, 256, 100000]
[0.003, 5e-05, [2048, 512, 128, 65], 1.0, 0.80927837, -4.61891, 0.5, 256, 100000]
[0.01, 5e-05, [2048, 512, 128, 65], 0.9921875, 0.78659797, -4.0049944, 0.5, 256, 100000]
[0.03, 5e-05, [2048, 512, 128, 65], 0.8828125, 0.7123711, -2.801302, 0.5, 256, 100000]
"""