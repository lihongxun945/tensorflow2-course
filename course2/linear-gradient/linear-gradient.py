import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('./data.csv')
x = data['age']
y = data['salary']
plt.scatter(x, y) # 画上原始数据

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(x, y, epochs=3000) # 对这个数据来说3000次左右即可

py = model.predict(x)
plt.scatter(x, py) # 画上预测数据
plt.show()