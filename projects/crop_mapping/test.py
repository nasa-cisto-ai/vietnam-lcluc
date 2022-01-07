import tensorflow as tf
import numpy as np

x = np.random.randint(1500, size=(24, 24, 4))

means = x.mean(axis=(0, 1))
stds = x.std(axis=(0, 1))

x = tf.convert_to_tensor(x)

x = tf.cast(x, tf.float32)
axis = (0, 1)
c = 1e-8
new = (x - tf.reduce_mean(x, axis=axis)) / (tf.math.reduce_std(x, axis=axis) + c)

print(x.shape)

new2 = list()
for i in range(x.shape[-1]):  # for each channel in images
    xx = (x[:, :, i] - means[i]) / (stds[i] + c)
    print(xx.shape)
    new2.append((x[:, :, i] - means[i]) / (stds[i] + c))
outputs = tf.stack(new2, axis=-1)
print(outputs.shape)


#print(means, stds, x, outputs)