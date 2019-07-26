import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from util import AttrDict

def build_bootstrapped_dnn(shape_x, shape_y, heads_n, learning_rate=0.001):
    '''
    Regression DNN with multiple 'bootstrap heads' for epistemic uncertainty estimation.

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

def build_bootstrapped_dnn_var(shape_x, shape_y, heads_n, learning_rate=0.001):
    '''
    Regression DNN with multiple 'bootstrap heads' and data variance estimation (aleatoric uncertainty).

    Based on "Deep Exploration via Bootstrapped DQN" by Ian Osband.
    http://papers.nips.cc/paper/6500-deep-exploration-via-bootstrapped-dqn

    Extended to estimate aleatoric uncertainty per head.
    
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
    
    means = []
    stds = []
    ys = []
    for k in range(heads_n):
        mean = tf.layers.dense(net, np.prod(shape_y), name='head-{}'.format(k))
        std = tf.layers.dense(net, np.prod(shape_y), tf.nn.softplus, name='head_std-{}'.format(k)) + 1e-6

        mean = tf.reshape(mean, [-1] + shape_y)
        std = tf.reshape(std, [-1] + shape_y)
        means.append(mean)
        stds.append(std)

        ys.append(y_pl) # also stack ys to have same shape as heads. Required for loss calculation.
        
    # create a single tensor containing all heads. Note: Shape is (heads_n, batch, shape_x)
    means = tf.stack(means, axis=0, name='means')
    stds = tf.stack(stds, axis=0, name='stds')
    ys = tf.stack(ys, axis=0, name='ys')
                    
    # mask active heads - transpose mask, as mask is (batch, heads_n) while heads and ys is (heads_n, batch, shape_x)
    means_masked = tf.boolean_mask(means, tf.transpose(bootstrap_mask_pl), name='means_masked')
    stds_masked = tf.boolean_mask(stds, tf.transpose(bootstrap_mask_pl), name='stds_masked')
    ys_masked = tf.boolean_mask(ys, tf.transpose(bootstrap_mask_pl), name='ys_masked')
    
    bootstrap_dist_masked = tfd.Independent(
        distribution=tfd.Normal(loc=means_masked, scale=stds_masked),
        reinterpreted_batch_ndims=1)

    # TODO is reduce_mean correct? tf.losses.mean_squared_error does sum_by_nonzero_weights
    loss = tf.math.reduce_mean(-bootstrap_dist_masked.log_prob(ys_masked))
    optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    data_dist = tfd.Independent(
        distribution=tfd.Normal(loc=means, scale=stds),
        reinterpreted_batch_ndims=1)
    
    # calc data mean and noise as mean of heads, epistemic uncertainty as stddev of head means
    data_mean = tf.math.reduce_mean(data_dist.mean(), axis=0)
    data_noise = tf.math.reduce_mean(data_dist.stddev(), axis=0)
    data_uncertainty = tf.math.reduce_std(data_dist.mean(), axis=0)

    return AttrDict(locals())

def build_dnn_var(shape_x, shape_y, batch_size, learning_rate=0.001):
    '''
    Regression DNN with data variance estimation (aleatoric uncertainty).

    
    args:
        shape_x: list, Shape of the input data (without batch dimension).
        shape_y: list, Shape of the labels (without batch dimenstion).
        learning_rate: float, learning rate.
    '''
    X_pl = tf.placeholder(shape=[None] + shape_x, dtype=tf.float32, name='X_pl')
    y_pl = tf.placeholder(shape=[None] + shape_y, dtype=tf.float32, name='y_pl')
    
    x = tf.layers.flatten(X_pl)
    net = tf.layers.dense(x, 128, activation=tf.nn.relu)
    net = tf.layers.dense(net, 128, activation=tf.nn.relu)
    
    mean = tf.layers.dense(net, np.prod(shape_y))
    std = tf.layers.dense(net, np.prod(shape_y), tf.nn.softplus) + 1e-6

    mean = tf.reshape(mean, [-1] + shape_y)
    std = tf.reshape(std, [-1] + shape_y)
    data_dist = tfd.Normal(loc=mean, scale=std)

    losses = [
        -data_dist.log_prob(y_pl),
    ]
    loss = sum(tf.reduce_sum(loss) for loss in losses) / tf.to_float(batch_size)

    optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    data_mean = data_dist.mean()
    data_noise = data_dist.stddev()

    return AttrDict(locals())