import tensorflow as tf
import os
from wgan_train import save_result

generator = tf.keras.models.load_model('./model.tf', compile=True)

z = tf.random.normal([100, 100])
fake_image = generator(z, training=False)
img_path = os.path.join('images', 'loaded.png')
save_result(fake_image.numpy(), 10, img_path, color_mode='P')