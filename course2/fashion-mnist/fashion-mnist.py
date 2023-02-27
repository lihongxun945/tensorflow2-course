from cgi import test
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
# 把每个像素点 0~255归一化为 0~1
train_image=train_image/255
test_image=test_image/255
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['acc'])
model.fit(train_image, train_label, epochs=10)
model.evaluate(test_image, test_label)