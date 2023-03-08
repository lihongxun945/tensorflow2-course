# 增加网络容量，增加拟合能力
import tensorflow as tf
import numpy as np

print('GPU', tf.config.list_physical_devices('GPU'))

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# 卷积神经网络的形状是4维度，分别是 batch,height,width,channel
# 因为这是一个灰度图片集，形状从 (60000, 28, 28) 变成 (60000, 28, 28, 1)
train_images = np.expand_dims(train_images, -1)
test_images= np.expand_dims(test_images, -1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu', padding='same'), # (None, 28, 28, 64)
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # (None, 28, 28, 64)
    tf.keras.layers.MaxPooling2D(), # (None, 14, 14, 64)
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'), # (None, 14, 14, 128)
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'), # (None, 14, 14, 128)
    tf.keras.layers.MaxPooling2D(), # (None, 7, 7, 128)
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'), # (None, 7, 7, 256)
    # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'), # (None, 7, 7, 256)
    tf.keras.layers.MaxPooling2D(), # (None, 3, 3, 256)
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'), # (None, 3, 3, 512)
    # tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'), # (None, 3, 3, 512)
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.GlobalAveragePooling2D(), # (None, 512) ，注意这里和flatten作用类似
    tf.keras.layers.Dense(256, activation='relu'), # (None, 256)
    tf.keras.layers.Dense(10, activation='softmax'), # (None, 10)
])
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))
