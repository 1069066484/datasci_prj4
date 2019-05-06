import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def variable_init(sz):
    in_dim = sz[0]
    w_stddev = 1.0 / tf.sqrt(in_dim / 2)
    return tf.random_normal(shape=size, stddev=w_stddev)


if __name__=="__main__":
    pass

