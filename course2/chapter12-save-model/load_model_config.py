import tensorflow as tf

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
# 把每个像素点 0~255归一化为 0~1
train_image=train_image/255
test_image=test_image/255
config_file = open('model_config.json')
json_config = config_file.read()
# 只有模型结构，后面还需要训练
print(json_config)
model = tf.keras.models.model_from_json(json_config)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(train_image, train_label, epochs=5)
model.evaluate(test_image, test_label)