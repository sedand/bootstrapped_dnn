import tensorflow as tf
import numpy as np
import models
import matplotlib.pyplot as plt

# Number of bootstrap heads
HEADS_N = 10

x_data = np.linspace(0,10,100)
y_data = np.sin(x_data) + np.random.normal(0, .2, x_data.shape)
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

# bootstrap mask - generate one for every element in x_data. Shape: (batch, heads_n)
x_data_mask = np.random.binomial(n=1, p=.1, size=(x_data.shape[0], HEADS_N))
print('Bootstrap mask:', x_data_mask)

tf.reset_default_graph()
# build graph
boot_dnn = models.build_bootstrapped_dnn(shape_x=list(x_data.shape[1:]), shape_y=list(y_data.shape[1:]), heads_n=HEADS_N)
print(boot_dnn)
# start session
sess = tf.InteractiveSession()
# init variables
sess.run(tf.global_variables_initializer())

# train
losses = []
for _ in range(1000):
    feed_dict = {boot_dnn.X_pl: x_data, boot_dnn.y_pl: y_data, boot_dnn.bootstrap_mask_pl: x_data_mask}
    _, loss = sess.run([boot_dnn.optimize, boot_dnn.loss], feed_dict)
    losses.append(loss)

# predict
x_test = np.linspace(-3, 13, 300).reshape(-1, 1)
feed_dict = {boot_dnn.X_pl: x_test}
heads, heads_mean, heads_var = sess.run([boot_dnn.heads, boot_dnn.heads_mean, boot_dnn.heads_var], feed_dict)

# flatten data from shape (batch,x_dim=1) to 1D in order plot
heads_mean = heads_mean.reshape(-1)
heads_var = heads_var.reshape(-1)

# note: heads is shape (heads, batch, x_dim)
print(heads.shape)

# plot
fig,axes = plt.subplots(nrows=2, figsize=(8,8))
axes[0].plot(losses)
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[1].fill_between(x_test.reshape(-1), heads_mean+heads_var, heads_mean-heads_var, color='r', alpha=.2, label="prediction var (epis)")
axes[1].fill_between(x_test.reshape(-1), heads_mean+heads_var*2, heads_mean-heads_var*2, color='r', alpha=.2)
axes[1].plot(x_test, heads_mean, '-', color='r', lw=1, label='prediction mean')
axes[1].set_ylim((2, -2))
for head in heads:
    axes[1].plot(x_test, head, color='r', lw=1, alpha=.2)
axes[1].scatter(x_data, y_data, color='b', s=1, label='training data')
plt.legend()
plt.savefig('bootstrap.png', bbox_inches='tight')
plt.show()