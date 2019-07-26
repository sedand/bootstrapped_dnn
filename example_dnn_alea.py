import tensorflow as tf
import numpy as np
import models
import matplotlib.pyplot as plt

NOISE = .2

x_data = np.linspace(0,10,500)
y_data = np.sin(x_data) + np.random.normal(0, NOISE, x_data.shape)
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

tf.reset_default_graph()
# build graph
boot_dnn = models.build_dnn_var(shape_x=list(x_data.shape[1:]), shape_y=list(y_data.shape[1:]), batch_size=x_data.shape[0])
print(boot_dnn)
# start session
sess = tf.InteractiveSession()
# init variables
sess.run(tf.global_variables_initializer())

# train
losses = []
for _ in range(2000):
    feed_dict = {boot_dnn.X_pl: x_data, boot_dnn.y_pl: y_data}
    _, loss = sess.run([boot_dnn.optimize, boot_dnn.loss], feed_dict)
    losses.append(loss)

# predict
x_test = np.linspace(-3, 13, 300).reshape(-1, 1)
feed_dict = {boot_dnn.X_pl: x_test}
data_mean, data_noise = sess.run([boot_dnn.data_mean, boot_dnn.data_noise], feed_dict)

# flatten data from shape (batch,x_dim=1) to 1D in order plot
data_mean = data_mean.reshape(-1)
data_noise = data_noise.reshape(-1)

print(data_mean.shape)

# plot
fig,axes = plt.subplots(nrows=3, figsize=(8,10))
axes[0].plot(losses)
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[1].fill_between(x_test.reshape(-1), data_mean+data_noise, data_mean-data_noise, color='r', alpha=.2, label="prediction std (noise/alea)")
axes[1].fill_between(x_test.reshape(-1), data_mean+data_noise*2, data_mean-data_noise*2, color='r', alpha=.2)
axes[1].plot(x_test, data_mean, '-', color='r', lw=1, label='prediction mean')
axes[1].set_ylim((2, -2))
axes[1].scatter(x_data, y_data, color='b', s=1, label='training data')
axes[1].legend()

axes[2].hlines(y=NOISE, xmin=min(x_data), xmax=max(x_data), color='r', label='True stddev')
axes[2].plot(x_test, data_noise, label='headmean data stddev (noise/alea)')
axes[2].legend()

plt.savefig('dnn_alea.pdf', bbox_inches='tight')
plt.show()