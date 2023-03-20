import tensorflow as tf

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
# 把每个像素点 0~255归一化为 0~1
train_image=train_image/255
test_image=test_image/255
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['acc'])
# 检查点只保存了weights
model.load_weights('cp')
model.evaluate(test_image, test_label)
