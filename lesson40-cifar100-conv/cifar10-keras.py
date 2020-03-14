# 课程上讲的是把网络分成两部分，然后手动计算了误差进行传播，原因是 卷积层到全连接层中间有一个形状转换
# 老师为了讲解写的比较细，其实只要加一个转换层，就可以搞定，不用手动计算那么多。

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)

# 加载数据
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# cifar100 下载较慢，你可以手动下载然后放到  ~/.keras/datasets 里面去
(x, y), (x_test, y_test) = datasets.cifar10.load_data()
print(x.shape, y.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(256)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(256)

# 这个模型大约是70%正确率
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)), # 16x16
    layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)), # 8x8
    layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)), # 4x4
    layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
    layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)), # 2x2

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])
model.summary()
model.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_db, epochs=50,
                    validation_data=test_db, validation_freq=2)