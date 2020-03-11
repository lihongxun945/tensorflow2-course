import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


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
db = db.map(preprocess).shuffle(10000).batch(128)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(128)

# 在 lesson30 中加了一行代码保存了模型，这里我们直接读出来
# model.save to save model
print('loaded model from file.')
network = tf.keras.models.load_model('model.h5', compile=True)
# 如果compile=false 那么就可以自己指定compile参数
#network.compile(optimizer=optimizers.Adam(lr=0.01),
#        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#        metrics=['accuracy']
#    )
network.evaluate(db_test)