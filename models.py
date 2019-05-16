import tensorflow as tf
import numpy as np
from util import AttrDict

def build_bootstrapped_dnn(shape_x, shape_y, heads_n, learning_rate=0.001):
    '''
    Regression DNN with multiple 'bootstrap heads'.

    Based on "Deep Exploration via Bootstrapped DQN" by Ian Osband.
    http://papers.nips.cc/paper/6500-deep-exploration-via-bootstrapped-dqn
    
    args:
        shape_x: list, Shape of the input data (without batch dimension).
        shape_y: list, Shape of the labels (without batch dimenstion).
        heads_n: int, Number of bootstrap heads (called K in the paper).
        learning_rate: float, learning rate.
    '''
    X_pl = tf.placeholder(shape=[None] + shape_x, dtype=tf.float32, name='X_pl')
    y_pl = tf.placeholder(shape=[None] + shape_y, dtype=tf.float32, name='y_pl')
    bootstrap_mask_pl = tf.placeholder(shape=[None] + [heads_n], dtype=tf.bool, name='bootstrap_mask_pl')
    
    x = tf.layers.flatten(X_pl)
    net = tf.layers.dense(x, 128, activation=tf.nn.relu)
    net = tf.layers.dense(net, 128, activation=tf.nn.relu)
    
    heads = []
    ys = []
    for k in range(heads_n):
        h = tf.layers.dense(net, np.prod(shape_y), name='head-{}'.format(k))
        h = tf.reshape(h, [-1] + shape_y)
        heads.append(h)
        ys.append(y_pl) # also stack ys to have same shape as heads. Required for loss calculation.
        
    # create a single tensor containing all heads. Note: Shape is (heads_n, batch, shape_x)
    heads = tf.stack(heads, axis=0, name='heads')
    ys = tf.stack(ys, axis=0, name='ys')
                    
    # mask active heads - transpose mask as mask is (batch, heads_n) while heads and ys is (heads_n, batch, shape_x)
    heads_masked = tf.boolean_mask(heads, tf.transpose(bootstrap_mask_pl), name='heads_masked')
    ys_masked = tf.boolean_mask(ys, tf.transpose(bootstrap_mask_pl), name='ys_masked')
    
    loss = tf.losses.mean_squared_error(heads_masked, ys_masked)
    optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    ## Calculate mean and variance over heads (dim 0)
    heads_mean, heads_var = tf.nn.moments(heads, axes=[0])

    return AttrDict(locals())