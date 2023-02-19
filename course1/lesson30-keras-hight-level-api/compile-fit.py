import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


# (x, y), (x_test, y_test) = datasets.fashion_mnist.load_data() # fashion mnist
(x, y), (x_test, y_test) = datasets.mnist.load_data() # 手写数字识别也是一样的
print(x.shape, y.shape)

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(256)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(256)

db_iter = iter(db)
sample = next(db_iter)

print('batch:', sample[0].shape, sample[1].shape)

network = Sequential([
    layers.Dense(256, activation=tf.nn.relu), #[b, 784] => [b, 256]
    layers.Dense(128, activation=tf.nn.relu),  # [b, 256] => [b, 128]
    layers.Dense(64, activation=tf.nn.relu),  # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu),  # [b, 64] => [b, 32]
    # layers.Dense(16, activation=tf.nn.relu),  # [b, 32] => [b, 16] # 尝试加一层
    layers.Dense(10, activation=tf.nn.relu),  # [b, 32] => [b, 10]
])

network.build(input_shape=[None, 28*28])
network.summary()

network.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.losses.MSE,
    metrics=['accuracy']
)

network.fit(db, epochs=10, validation_data=db_test, validation_freq=2)
network.save('./model.h5')