# 课程上讲的是把网络分成两部分，然后手动计算了误差进行传播，原因是 卷积层到全连接层中间有一个形状转换
# 老师为了讲解写的比较细，其实只要加一个转换层，就可以搞定，不用手动计算那么多。

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)

# 加载数据
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# cifar100 下载较慢，你可以手动下载然后放到  ~/.keras/datasets 里面去
(x, y), (x_test, y_test) = datasets.cifar100.load_data()

print(x.shape, y.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

# 卷积层取特征
# maxpool层强化特征并且把图片尺寸减小一半
network = Sequential([
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"), # 16x16

    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"), # 8x8

    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"), # 4x4

    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"), # 2x2

    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"), # 1x1

    # 转换形状
    layers.Reshape((-1, 512), input_shape=(-1, 1, 1, 512)), # 这里加一个 Reshape层就好啦

    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(100, activation=None),
])

network.build(input_shape=[None, 32, 32, 3])
network.summary()

# 用 keras 的高层API直接训练
network.compile(
    optimizer=optimizers.Adam(lr=1e-4),
    loss=tf.losses.categorical_crossentropy, # MSE 是个对象， CategoricalCrossentropy 是个类
    metrics=['accuracy']
)

network.fit(train_db, epochs=10, validation_data=test_db, validation_freq=2)
network.save('./model.h5')