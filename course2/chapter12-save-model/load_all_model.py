import tensorflow as tf

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
# 把每个像素点 0~255归一化为 0~1
test_image=test_image/255

# 不需要模型代码，都已经保存了
# h5格式
model = tf.keras.models.load_model('./all_model.h5')
model.evaluate(test_image, test_label)

# tf格式
model = tf.keras.models.load_model('./all_model')
model.evaluate(test_image, test_label)
