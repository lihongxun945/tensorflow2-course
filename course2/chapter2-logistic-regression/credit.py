import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('./credit-a.csv')
x = data.iloc[:, :-1]
y = data.iloc[:, -1].replace(-1, 0) # 需要在0~1之間

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(15,), activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) ## metrics 是在训练过程中输出准确率，比loss直观
model.summary()
history = model.fit(x, y, epochs=200)

plt.plot(history.epoch, history.history.get('acc'), history.history.get('loss'))
plt.show()