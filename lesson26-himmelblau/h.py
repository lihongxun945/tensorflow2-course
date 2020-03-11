import numpy as np
import tensorflow as tf

def h(x):
    return (x[0]**2+x[1]-1)**2 + (x[0] + x[1]**2 -7)**2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x, y shape:', x.shape, y.shape)

x = tf.Variable(tf.constant([4., 0.]))

for step in range(100):
    with tf.GradientTape() as tape:
        y = h(x)

    grads = tape.gradient(y, [x])[0]

    x.assign_sub(0.01 * grads)

    if step % 10 == 0:
        print('step {}: x={}, y={}'.format(step, x.numpy(), y.numpy()))