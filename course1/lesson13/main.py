import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# x: [60k, 28, 28]
# y: [60k]
# x_test: [10k, 28, 28]
# y_test: [60k]

(x, y), (x_test, y_test) = datasets.mnist.load_data()

x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_min(y))

train_db = tf.data.Dataset.from_tensor_slices(((x, y))).batch(128)
test_db = tf.data.Dataset.from_tensor_slices(((x_test, y_test))).batch(128)


# [b, 784] => [b, 256], => [b, 128] => [b, 10]

w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

for epoch in range(100):
    for step, (x, y) in enumerate(train_db):
        # x:[128, 28, 28]
        # y:[128]

        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28*28])

        lr = 0.001

        with tf.GradientTape() as tape:
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784] @[784, 256] + [256] => [b ,256]
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3

            # compute loss
            # out: [b, 10]
            # y: [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10) # one hot 是把0~9的值变成一个向量表示，比如 1 变成 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

            # mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_onehot - out))

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print('loss:', loss)

    total_correct = 0
    total_num = 0

    for step, (x, y) in enumerate(test_db):
        # x:[128, 28, 28]
        # y:[128]

        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28*28])

        with tf.GradientTape() as tape:
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784] @[784, 256] + [256] => [b ,256]
            h1 = tf.nn.relu(x@w1 + b1)
            h2 = tf.nn.relu(h1@w2 + b2)
            out = h2@w3+b3

            # out: [b, 10] ~ R
            # prob: [b, 10] ~ [0, 1]
            # y: [b]
            prob = tf.nn.softmax(out, axis=1) # 把0~9转成 0~1 范围内
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32) # 求一个最大值，默认是int64

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_correct += int(correct)
            total_num += x.shape[0]

    print(total_correct, total_num, total_correct/total_num)