import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()

        # 这里解释下padding=same的规则，他是在输入上做padding的
        # 如果stride不是1，那么他会通过补全输入的维度，使得输出是你预计的值

        # 第一层，如果stride不是1，那么其实进行了一次采样，会导致输出的结果形状比输入小
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # 第二层就不要下采样了，不然越来越小
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

        # 如果短接了，那么也要保证结果的shape是一样的
        if strides != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=strides))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):
        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out, training=training)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        # layers.add 就是简单的把两个张量加到一起
        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=100): # layer_dims: [2, 2, 2, 2], num_classes 就是输出结果数
        super(ResNet, self).__init__()

        # 根节点，对数据进行预处理，不是一个resblock
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        # 四层 resblock
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], strides=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], strides=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], strides=2)

        # 输出层，到这一层处理前，形状是 [b, 512, h, w]，
        # 我们经过3次strides=2 处理，结果形状应该被缩小到 1/8，如果输入是 (32, 32) 那么到这里就是 (4,4)，而我们需要h w变成 (1, 1),
        # 因此加一个平均层，无论最后(h,w)值如何，都会被取一个平均值而缩减到(1,1)，比如如果是 (4,4) 那么就取所有16个值的平均值
        self.avgpool = layers.Flatten() #
        self.fc = layers.Dense(num_classes) # 全连接层输出结果

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    # 一个resblock 中间包含了n个basicblock
    def build_resblock(self, filter_num, blocks, strides=1):

        res_block = Sequential()
        res_block.add(BasicBlock(filter_num, strides=strides))

        for _ in range(1, blocks):
            res_block.add(BasicBlock(filter_num, strides=1))

        return res_block

def resnet10():
    return ResNet([1, 1, 1, 1])

def resnet18():
    return ResNet([2, 2, 2, 2])

def resnet34():
    return ResNet([3, 4, 6, 3])

def resnet48():
    return ResNet([4, 4, 4, 4, 4, 2])