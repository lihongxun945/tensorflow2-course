# 课程上讲的是把网络分成两部分，然后手动计算了误差进行传播，原因是 卷积层到全连接层中间有一个形状转换
# 老师为了讲解写的比较细，其实只要加一个转换层，就可以搞定，不用手动计算那么多。

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from resnet2 import ResNet

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)

# 加载数据
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255. -1
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# cifar100 下载较慢，你可以手动下载然后放到  ~/.keras/datasets 里面去
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
print(x.shape, y.shape)
y = tf.squeeze(y, axis=1) # [n, 1] => [n]
y_test = tf.squeeze(y_test, axis=1) # [n, 1] => [n]
print(x.shape, y.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(256)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(256)

network = ResNet(100)

network.build(input_shape=(None, 32, 32, 3))
network.summary()

# 用 keras 的高层API直接训练
network.compile(
    optimizer=optimizers.Adam(lr=1e-4),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True), # MSE 是个对象， CategoricalCrossentropy 是个类
    metrics=['accuracy']
)

network.fit(train_db, epochs=20, validation_data=test_db, validation_freq=2)