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
            labeled_data = list(labeled_data)
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
        #return tf.Variable(tf.truncated_normal([in_sz, out_sz],stddev=0.1))
        return tf.random_uniform(w_shape,-0.5,0.5)

    @staticmethod
    def _b_generator(b_shape):
        return tf.random_uniform(b_shape,-0.5,0.5)

    def _add_layer(self, input, in_sz, out_sz, act_fun=None):
        #w = tf.Variable(tf.truncated_normal([in_sz, out_sz],stddev=0.1))
        w = tf.Variable(self.w_generator(w_shape=[in_sz,out_sz]))
        self._l2_loss += tf.nn.l2_loss(w) * self._l2_reg
        #tf.assign_add(self._l2_loss, tf.nn.l2_loss(w) * self._l2_reg)
        b = tf.Variable(self.b_generator(b_shape=[out_sz]))

        wx_plusb = tf.matmul(input, w) + b
        output = wx_plusb if act_fun is None else act_fun(wx_plusb)
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
            h = add_layer(h, prev_dim, neurons, act_fun=tf.nn.sigmoid)
            prev_dim = neurons
        #h1 = add_layer(self._x, self._input_dim, 64, act_fun=tf.nn.relu)
        #h2 = add_layer(h1, 64, 32, act_fun=tf.nn.relu)
        self._yo = add_layer(h, prev_dim, self._label_count, act_fun=tf.nn.sigmoid)
        
        #loss = -tf.reduce_sum(self._y*tf.log(self._yo))
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._yo))
        #loss = tf.reduce_sum(tf.square(self._y - self._yo)) 
        self._trainer = tf.train.AdamOptimizer(self._opt_step).minimize(self._loss + self._l2_loss)
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
                to_prt[-1] += [classifier.neurons(), hist[0][-1], hist[1][-1], hist[2][-1]]
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
    for l2 in [0.0]:
        for opt_step in [0.005]:
            for h_neurons in GeneralNNClassifier.neuron_arrange(neurons, 3):
                to_prt.append([l2, opt_step])
                print('\n\n',l2, opt_step, h_neurons)
                classifier = GeneralNNClassifier(os.path.join(global_defs.PATH_SAVING, 'NN_classifier_test'), labeled_data=dl_src, opt_step=opt_step, l2=l2, h_neurons=h_neurons)
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
    neurons = [[196, 128, 96], [196, 128, 128], [128,128,96], [128,96,96], [96,96,96], [96, 96, 80], [96, 80, 80], [80,80,80]]
    to_prt = []
    for l2 in [0.0]:
        for opt_step in [0.005]:
            #for h_neurons in GeneralNNClassifier.neuron_arrange(neurons, 3):
            for h_neurons in neurons:
                to_prt.append([l2, opt_step])
                print('\n\n',l2, opt_step, h_neurons)
                classifier = GeneralNNClassifier(os.path.join(global_defs.PATH_SAVING, 'NN_classifier_test'), labeled_data=dl_src, opt_step=opt_step, l2=l2, h_neurons=h_neurons)
                classifier.train(batch_sz=64, iterations=3000)
                hist = classifier.history()
                to_prt[-1] += [classifier.neurons(), hist[0][-1], hist[1][-1], hist[2][-1], classifier._train_keep_prob]
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
    _main_DA_test()

"""
l2  opt_step    nns   train_acc   test_acc    loss  
[0.0, 0.001, [784, 128, 128, 128, 10], 0.9375, 0.9076, 1.5378909]
[0.0, 0.001, [784, 128, 128, 256, 10], 0.890625, 0.9025, 1.5612679]
[0.0, 0.001, [784, 128, 128, 512, 10], 0.59375, 0.5118, 1.8742998]
[0.0, 0.001, [784, 128, 256, 256, 10], 0.890625, 0.9091, 1.5649452]
[0.0, 0.001, [784, 128, 256, 512, 10], 0.84375, 0.833, 1.594212]
[0.0, 0.001, [784, 128, 512, 512, 10], 0.890625, 0.9009, 1.5698087]
[0.0, 0.001, [784, 256, 256, 256, 10], 0.921875, 0.9155, 1.5444417]
[0.0, 0.001, [784, 256, 256, 512, 10], 0.78125, 0.7569, 1.6647973]
[0.0, 0.001, [784, 256, 512, 512, 10], 0.90625, 0.9024, 1.5472938]
[0.0, 0.001, [784, 512, 512, 512, 10], 0.890625, 0.8461, 1.5894291]

[0.0, 0.001, [784, 128, 128, 128, 10], 0.921875, 0.9013, 1.5528482]
[0.0, 0.001, [784, 256, 128, 128, 10], 0.921875, 0.9212, 1.547314]
[0.0, 0.001, [784, 512, 128, 128, 10], 0.90625, 0.9239, 1.5390812]
[0.0, 0.001, [784, 256, 256, 128, 10], 0.90625, 0.92, 1.5350097]
[0.0, 0.001, [784, 512, 256, 128, 10], 0.921875, 0.9214, 1.5338281]
[0.0, 0.001, [784, 512, 512, 128, 10], 0.90625, 0.9224, 1.5401504]
[0.0, 0.001, [784, 256, 256, 256, 10], 0.921875, 0.919, 1.5426128]
[0.0, 0.001, [784, 512, 256, 256, 10], 0.921875, 0.9181, 1.5332631]
[0.0, 0.001, [784, 512, 512, 256, 10], 0.890625, 0.9209, 1.5360472]
[0.0, 0.001, [784, 512, 512, 512, 10], 0.875, 0.8404, 1.5791806]

[0.001, 0.001, [784, 128, 128, 128, 10], 0.78125, 0.7592, 1.7162681]
[0.001, 0.001, [784, 256, 128, 128, 10], 0.78125, 0.7869, 1.7015862]
[0.001, 0.001, [784, 512, 128, 128, 10], 0.78125, 0.7547, 1.7107595]
[0.001, 0.001, [784, 256, 256, 128, 10], 0.828125, 0.8201, 1.6745846]
[0.001, 0.001, [784, 512, 256, 128, 10], 0.84375, 0.8072, 1.6829329]
[0.001, 0.001, [784, 512, 512, 128, 10], 0.703125, 0.7183, 1.732008]
[0.001, 0.001, [784, 256, 256, 256, 10], 0.703125, 0.739, 1.7358925]
[0.001, 0.001, [784, 512, 256, 256, 10], 0.8125, 0.7785, 1.6942048]
[0.001, 0.001, [784, 512, 512, 256, 10], 0.828125, 0.8233, 1.6828978]
[0.001, 0.001, [784, 512, 512, 512, 10], 0.3125, 0.2891, 2.1255116]

l2  opt_step    nns   train_acc   test_acc    loss   _train_keep_prob
DA.A2R  pos_neurons= [128, 348, 1024, 128, 348, 1024]
[0.0, 0.005, [2048, 1024, 1024, 348, 65], 0.09375, 0.036082473, 4.163493, 0.5]
[0.0, 0.005, [2048, 1024, 1024, 128, 65], 0.046875, 0.035051547, 4.1359577, 0.5]
[0.0, 0.005, [2048, 1024, 348, 348, 65], 0.125, 0.054639176, 4.1294937, 0.5]
[0.0, 0.005, [2048, 1024, 348, 128, 65], 0.046875, 0.07113402, 4.012774, 0.5]
[0.0, 0.005, [2048, 1024, 128, 128, 65], 0.09375, 0.10618557, 3.9326963, 0.5]
[0.0, 0.005, [2048, 348, 348, 128, 65], 0.28125, 0.21030928, 3.6706357, 0.5]
[0.0, 0.005, [2048, 348, 128, 128, 65], 0.234375, 0.21340206, 3.7003455, 0.5]

DA.A2R  pos_neurons= [128, 384, 512, 768, 128, 384, 512, 768]
[0.0, 0.005, [2048, 768, 768, 512, 65], 0.03125, 0.03814433, 4.173172, 0.5]
[0.0, 0.005, [2048, 768, 768, 384, 65], 0.03125, 0.041237112, 4.174387, 0.5]
[0.0, 0.005, [2048, 768, 768, 128, 65], 0.0625, 0.0927835, 3.9597526, 0.5]
[0.0, 0.005, [2048, 768, 512, 512, 65], 0.09375, 0.06082474, 4.100288, 0.5]
[0.0, 0.005, [2048, 768, 512, 384, 65], 0.078125, 0.03814433, 4.082363, 0.5]
[0.0, 0.005, [2048, 768, 512, 128, 65], 0.09375, 0.074226804, 3.8893306, 0.5]
[0.0, 0.005, [2048, 768, 384, 384, 65], 0.09375, 0.073195875, 4.0264263, 0.5]
[0.0, 0.005, [2048, 768, 384, 128, 65], 0.046875, 0.08865979, 3.9570823, 0.5]
[0.0, 0.005, [2048, 768, 128, 128, 65], 0.171875, 0.08865979, 3.7726479, 0.5]
[0.0, 0.005, [2048, 512, 512, 384, 65], 0.09375, 0.072164945, 4.015624, 0.5]
[0.0, 0.005, [2048, 512, 512, 128, 65], 0.109375, 0.09072165, 3.9183712, 0.5]
[0.0, 0.005, [2048, 512, 384, 384, 65], 0.015625, 0.025773196, 4.1743875, 0.5]
[0.0, 0.005, [2048, 512, 384, 128, 65], 0.0625, 0.11030928, 3.9769406, 0.5]
[0.0, 0.005, [2048, 512, 128, 128, 65], 0.296875, 0.17525773, 3.6032736, 0.5]
[0.0, 0.005, [2048, 384, 384, 128, 65], 0.109375, 0.087628864, 3.8431854, 0.5]
[0.0, 0.005, [2048, 384, 128, 128, 65], 0.09375, 0.18453608, 3.7436414, 0.5]

DA.A2R  pos_neurons= [96, 196, 384, 96, 196, 384]
[0.0, 0.005, [2048, 384, 384, 196, 65], 0.09375, 0.09690721, 4.0001626, 0.5]
[0.0, 0.005, [2048, 384, 384, 96, 65], 0.125, 0.14432989, 3.7490792, 0.5]
[0.0, 0.005, [2048, 384, 196, 196, 65], 0.15625, 0.12474227, 3.8255582, 0.5]
[0.0, 0.005, [2048, 384, 196, 96, 65], 0.140625, 0.10103093, 3.729929, 0.5]
[0.0, 0.005, [2048, 384, 96, 96, 65], 0.21875, 0.13195877, 3.698276, 0.5]
[0.0, 0.005, [2048, 196, 196, 96, 65], 0.171875, 0.20515464, 3.7365036, 0.5]
[0.0, 0.005, [2048, 196, 96, 96, 65], 0.328125, 0.27216494, 3.4658968, 0.5]
"""