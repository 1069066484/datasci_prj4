import numpy as np
import Ldata_helper as data_helper
import tensorflow as tf
import Lglobal_defs as global_defs
import os


class GeneralNNClassifier:
    def __init__(self, saving_path, labeled_data=None, h_neurons=[4096, 1024, 256], train_test_split=0.6, train_dl=None, test_dl=None):
        self._saving_path = saving_path
        if labeled_data is not None:
            labeled_data[1] = data_helper.labels2one_hot(labeled_data[1])
            [train_dl, test_dl] = data_helper.labeled_data_split(labeled_data, train_test_split)
        else:
            self._data, self._labels = train_dl
            test_dl = test_dl
        self._t_ld = test_dl
        self._labels = train_dl[1]
        self._data = train_dl[0]
        self._input_dim = self._data.shape[1]
        self._label_count = len(np.unique(self._labels))
        self._curr_batch_idx = 0

    @staticmethod
    def _gen_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def _gen_bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def _add_layer(self, input, in_sz, out_sz, act_fun=None):
        w = tf.Variable(tf.random_normal([in_sz, out_sz]))
        b = tf.Variable(tf.zeros([1, out_sz]) + 0.1)
        wx_plusb = tf.matmul(input, w) + b
        output = wx_plusb if act_fun is None else act_fun(wx_plusb)
        return tf.nn.dropout(output, self._keep_prob)

    def _construct_nn(self):
        add_layer = self._add_layer
        self._keep_prob = tf.placeholder(tf.float32)
        self._x = tf.placeholder(tf.float32, [None, self._input_dim])
        self._y = tf.placeholder(tf.float32, [None, self._label_count])
        h1 = add_layer(self._x, self._input_dim, 1024, act_fun=tf.nn.relu)
        h2 = add_layer(h1, 1024, 256, act_fun=tf.nn.relu)
        self._yo = add_layer(h2, 256, self._label_count, act_fun=tf.nn.relu)
        cross_entropy = -tf.reduce_sum(self._y*tf.log(self._yo))
        self._trainer = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self._yo, 1), tf.argmax(self._y, 1))
        self._acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _next_batch(self, batch_sz):
        indices = list(range(self._curr_batch_idx, self._curr_batch_idx+batch_sz))
        indices = [i-self._data.shape[0] if i >= self._data.shape[0] else i for i in indices]
        return [self._data[indices], self._labels[indices]]
    
    def _eval(self, labeled_data):
        return self._acc.eval(feed_dict={self._x: labeled_data[0], self._y:labeled_data[1], self._keep_prob:1.0})

    def train(self, iterations=10000, batch_sz=50):
        self._construct_nn()
        self._sess = tf.InteractiveSession()
        for i in range(iterations):
            data, labels = self._next_batch(batch_sz)
            if i % 300 == 0:
                print('it={}'.format(i), '  train_acc=', self._eval([data, labels]), '  test_acc=', self._eval(self._t_ld))
            self._trainer.run(feed_dict={x:data, y_:labels, self._keep_prob:0.5})


if __name__=='__main__':
    mnist = data_helper.read_mnist(one_hot=True)
    print('\n\n\n\n\n',mnist.test.images.shape, mnist.test.labels.shape) #(10000, 784) (10000, 10)
    print('\n\n\n\n\n',mnist.train.images.shape, mnist.train.labels.shape)#(55000, 784) (55000, 10)
    classifier = GeneralNNClassifier(os.path.join(global_defs.PATH_SAVING, 'NN_classifier_test'), train_dl=[mnist.train.images, mnist.train.labels], test_dl=[mnist.test.images, mnist.test.labels])
    classifier.train()
