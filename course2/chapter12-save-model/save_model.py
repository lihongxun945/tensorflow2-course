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
# 训练过程中保存
cp = tf.keras.callbacks.ModelCheckpoint('cp', save_weights_only=True)
history = model.fit(train_image, train_label, epochs=5, callbacks=[cp])
model.evaluate(test_image, test_label)

# 保存模型结构、权重、优化器的等所有数据
# h5格式，单文件
model.save('./all_model.h5')
# tf 格式，多文件
model.save('./all_model', save_format='tf')

# 仅保存模型结构
json_config = model.to_json()
json_config_file = open('model_config.json', 'w')
json_config_file.write(json_config)

# 仅保存模型权重
# weights = model.get_weights()
model.save_weights('weights.h5')
