import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('./Advertising.csv')
# plt.scatter(data['TV'], data['sales'])
# plt.scatter(data['radio'], data['sales'])
# plt.scatter(data['newspaper'], data['sales'])
# plt.show()
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3,), activation="relu"),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(x, y, epochs=2000)

py = model.predict(x)
print(py)