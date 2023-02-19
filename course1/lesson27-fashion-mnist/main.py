import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# (x, y), (x_test, y_test) = datasets.fashion_mnist.load_data() # fashion mnist
(x, y), (x_test, y_test) = datasets.mnist.load_data() # 手写数字识别也是一样的
print(x.shape, y.shape)

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(128)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(128)

db_iter = iter(db)
sample = next(db_iter)

print('batch:', sample[0].shape, sample[1].shape)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu), #[b, 784] => [b, 256]
    layers.Dense(128, activation=tf.nn.relu),  # [b, 256] => [b, 128]
    layers.Dense(64, activation=tf.nn.relu),  # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu),  # [b, 64] => [b, 32]
    layers.Dense(10, activation=tf.nn.relu),  # [b, 32] => [b, 10]
])

model.build(input_shape=[None, 28*28])
model.summary()

# w = w - lr*grad
optimizer = optimizers.Adam(lr=1e-3)

def main():
    for epoch in range(20):
        for step, (x, y) in enumerate(db):

            # x: [b, 28, 28] => [b, 28*28]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                # [b]
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))

            grads = tape.gradient(loss_mse, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, ', losses:', float(loss_mse))

        total_correct = 0
        total_num = 0
        for x, y in db_test:
            # x: [b, 28, 28] => [b, 28*28]
            # y: [b]
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x) # logits = [b, 10]
            prob = tf.nn.softmax(logits, axis=1) # [b, 10] => [b, int64]

            pred = tf.argmax(prob, axis=1) # 返回指定维度中最大值的index，降了一维
            pred = tf.cast(pred, dtype=tf.int32)

            total_correct += int(tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32)))
            total_num += x.shape[0]

        print(epoch, 'test accracy:', total_correct/total_num)

main()