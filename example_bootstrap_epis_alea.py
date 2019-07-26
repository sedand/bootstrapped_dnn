import tensorflow as tf
import numpy as np
import models
import matplotlib.pyplot as plt

# Number of bootstrap heads
HEADS_N = 10
BOOT_P = .2
NOISE = .2

x_data = np.linspace(0,10,500)
y_data = np.sin(x_data) + np.random.normal(0, NOISE, x_data.shape)
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

# bootstrap mask - generate one for every element in x_data. Shape: (batch, heads_n)
x_data_mask = np.random.binomial(n=1, p=BOOT_P, size=(x_data.shape[0], HEADS_N))
print('Bootstrap mask:', x_data_mask)

tf.reset_default_graph()
# build graph
boot_dnn = models.build_bootstrapped_dnn_var(shape_x=list(x_data.shape[1:]), shape_y=list(y_data.shape[1:]), heads_n=HEADS_N)
print(boot_dnn)
# start session
sess = tf.InteractiveSession()
# init variables
sess.run(tf.global_variables_initializer())

# train
losses = []
for _ in range(2000):
    feed_dict = {boot_dnn.X_pl: x_data, boot_dnn.y_pl: y_data, boot_dnn.bootstrap_mask_pl: x_data_mask}
    _, loss = sess.run([boot_dnn.optimize, boot_dnn.loss], feed_dict)
    losses.append(loss)

# predict
x_test = np.linspace(-3, 13, 300).reshape(-1, 1)
feed_dict = {boot_dnn.X_pl: x_test}
data_mean, data_noise, data_uncertainty, head_means = sess.run([boot_dnn.data_mean, boot_dnn.data_noise, boot_dnn.data_uncertainty, boot_dnn.data_dist.mean()], feed_dict)

# flatten data from shape (batch,x_dim=1) to 1D in order plot
data_mean = data_mean.reshape(-1)
data_noise = data_noise.reshape(-1)
data_uncertainty = data_uncertainty.reshape(-1)

# plot
fig,axes = plt.subplots(nrows=5, figsize=(8,12))
axes[0].plot(losses)
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[1].fill_between(x_test.reshape(-1), data_mean+data_uncertainty, data_mean-data_uncertainty, color='r', alpha=.2, label="prediction uncertainty stddev (epis)")
axes[1].fill_between(x_test.reshape(-1), data_mean+data_uncertainty*2, data_mean-data_uncertainty*2, color='r', alpha=.2)
axes[1].plot(x_test, data_mean, '-', color='r', lw=1, label='prediction mean')
axes[1].set_ylim((2, -2))
for head in head_means:
    axes[1].plot(x_test, head, color='r', lw=1, alpha=.2)
axes[1].scatter(x_data, y_data, color='b', s=1, label='training data')
axes[1].legend()

axes[2].plot(x_test, data_uncertainty, label='prediction uncertainty stddev')
axes[2].legend()

axes[3].fill_between(x_test.reshape(-1), data_mean+data_noise, data_mean-data_noise, color='r', alpha=.2, label="headmean data stddev (noise/alea)")
axes[3].fill_between(x_test.reshape(-1), data_mean+data_noise*2, data_mean-data_noise*2, color='r', alpha=.2)
axes[3].plot(x_test, data_mean, '-', color='r', lw=1, label='prediction mean')
axes[3].set_ylim((2, -2))
for head in head_means:
    axes[3].plot(x_test, head, color='r', lw=1, alpha=.2)
axes[3].scatter(x_data, y_data, color='b', s=1, label='training data')
axes[3].legend()

axes[4].hlines(y=NOISE, xmin=min(x_data), xmax=max(x_data), color='r', label='True stddev')
axes[4].plot(x_test, data_noise, label='headmean data stddev (noise/alea)')
axes[4].legend()

plt.savefig('bootstrap_epis_alea.pdf', bbox_inches='tight')
plt.show()