# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import glob
import random

all_image_path = glob.glob('./images/*/*.jpg')
random.shuffle(all_image_path)
print(all_image_path)
label_to_index = {'airplane': 0,  'lake': 1}
index_to_label = dict((v, k) for k, v in label_to_index.items())
all_labels = [label_to_index.get(p.split('/')[-1].split('_')[0]) for p in all_image_path]
print(all_labels)

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image/255.0  # normalize to [0,1] range
    return image

img_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
img_ds = img_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(all_labels)
image_label_ds = tf.data.Dataset.zip((img_ds, label_ds))
print(image_label_ds)

image_count = len(all_image_path)
test_count = int(image_count*0.2)
train_count = image_count - test_count
train_ds = image_label_ds.skip(test_count)
test_ds = image_label_ds.take(test_count)
BATCH_SIZE = 16
train_ds = train_ds.repeat().shuffle(buffer_size=train_count).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

model = tf.keras.Sequential()   #顺序模型
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['acc']
)

steps_per_epoch = train_count//BATCH_SIZE
validation_steps = test_count//BATCH_SIZE

history = model.fit(train_ds, epochs=10, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_data=test_ds, 
                    validation_steps=validation_steps)