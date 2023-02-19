import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # [b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => [b, 64, 64, 3]

        self.fc = layers.Dense(3*3*512) # 先把 100 个feature变成 3*3*512个

        # 参数分别是 filters = 256, kernal_size=3, stride=3, padding='valid'
        # [b, 3, 3, 512] => [b, 9, 9, 256]
        # 反卷积计算公式 W=(N−1)∗S−2P+F
        # 这里一步计算尺寸就是 (3-1)*3+3 = 9
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()

        # [b, 9, 9, 256] => [b, 21, 21, 128] ，计算尺寸是 (9-1)*2+5 = 21
        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        # [b, 21, 21, 128] => [b, 64, 64, 3] ，计算尺寸是 (21-1)*3+4 = 64
        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):
        # inputs [b, 100] = > [b, 3*3*512]
        x = self.fc(inputs)

        # [b, 3*3*512] => [b, 3, 3, 512]
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        x = tf.tanh(x)
        return x

# 判别器是一个姜维过程，把图片变成 0|1
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # [b, 64, 64, 3] => [b, 1]
        # 把 [64, 64, 3] 的图片，变成 0 | 1

        # [b, 64, 64, 3] => [b, 20, 20, 64]
        self.conv1 = layers.Conv2D(64, 5, 3, 'valid') # 四个参数分别是 filters=64, kernel_size=5, strides=3, padding='valid'

        # [b, 20, 20, 1] => [b, 6, 6, 128]
        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        # [b, 6, 6, 1] => [b, 1, 1, 256]
        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        # [b, h, w, c] => [b, -1]
        self.flatten = layers.Flatten()

        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))

        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        x = self.flatten(x)

        # [b, -1] => [b, 1]
        logits = self.fc(x)

        return logits

def main():
    gpu = tf.config.list_physical_devices('GPU')
    if len(gpu) > 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    d = Discriminator()
    g = Generator()

    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])

    prob = d(x)
    print(prob.shape)

    x_hat = g(z)
    print(x_hat.shape)

if __name__ == '__main__':
    main()