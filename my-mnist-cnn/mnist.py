# 试试别人的网络 https://www.zhihu.com/question/52893753

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
# 加载数据
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28, 28, 1])
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.mnist.load_data()

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

# 卷积层取特征
# maxpool层强化特征并且把图片尺寸减小一半
network = Sequential([
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

network.build(input_shape=[None, 28, 28, 1])
network.summary()

# 用 keras 的高层API直接训练
network.compile(
    optimizer=optimizers.Adam(lr=1e-4),
    loss=tf.losses.MSE,
    metrics=['accuracy']
)

network.fit(train_db, epochs=10, validation_data=test_db, validation_freq=2)