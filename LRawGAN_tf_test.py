import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Ldata_helper as data_helper
import Lglobal_defs as global_defs
import os


def variable_init(sz):
    in_dim = sz[0]
    w_stddev = 1.0 / tf.sqrt(in_dim / 2)
    return tf.random_normal(shape=sz, stddev=w_stddev)


def sample_uniform(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])


def plot(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape((28,28)), cmap='Greys_r')
    return fig


def main():
    X = tf.placeholder(tf.float32, shape=[None, 784])
    D_W1 = tf.Variable(variable_init([784,128]))
    D_b1 = tf.Variable(tf.zeros(shape=[128]))
    D_W2 = tf.Variable(variable_init([128,1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    theta_D = [D_W1, D_W2, D_b1, D_b2]

    Z = tf.placeholder(tf.float32, shape=[None, 100])
    G_W1 = tf.Variable(variable_init([100,128]))
    G_b1 = tf.Variable(tf.zeros([128]))
    G_W2 = tf.Variable(variable_init([128,784]))
    G_b2 = tf.Variable(tf.zeros([784]))
    theta_G = [G_W1, G_W2, G_b1, G_b2]

    def _generator(z):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

    def _discriminator(x):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    G_sample = _generator(Z)
    D_real, D_logit_real = _discriminator(X)
    D_fake, D_logit_fake = _discriminator(G_sample)
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    mb_size, Z_dim = 128, 100
    mnist = data_helper.read_mnist(one_hot=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dir_out = 'Lsaving/out'
    global_defs.mk_dir(dir_out)
    i = 0
    for it in range(20000):
        if it % 2000 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_uniform(16, Z_dim)})
            fig = plot(samples)
            plt.savefig(os.path.join(dir_out,'{}.png'.format(str(i).zfill(3))), bbox_inches='tight')
            i += 1
            plt.close(fig)
        X_mb, _ = mnist.train.next_batch(mb_size)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_uniform(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_uniform(mb_size, Z_dim)})
        if it % 2000 == 0:
            print('iter: ',it)
            print('D_loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))


if __name__=="__main__":
    main()

