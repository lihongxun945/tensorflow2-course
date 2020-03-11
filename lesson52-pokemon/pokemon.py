import os, glob
import random, csv

import tensorflow as tf

join = os.path.join

# root: 根目录
# filename： csv的文件名
# name2label: 编码表， 目录名->数字
def load_csv(root, filename, name2label):
    if not os.path.exists(join(root, filename)):
        print('creating csv file...')
        images = []
        for name in name2label.keys():
            images += glob.glob(join(root, name, '*.jpg'))
            images += glob.glob(join(root, name, '*.png'))
            images += glob.glob(join(root, name, '*.jpeg'))

        random.shuffle(images)

        with open(join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                label = name2label[name]
                # 'pokemon/pikachu/0001.png', 1
                writer.writerow([img, label])

    images, labels = [], []
    print('generating data...')
    with open(join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label)
            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)
    return images, labels

def load_pokemon(root, mode="train"):
    name2label = {}
    for name in sorted(os.listdir(join(root))):
        if os.path.isdir(join(root, name)):
            name2label[name] = len(name2label.keys())

    images, labels = load_csv(root, 'image.csv', name2label)

    length = len(images)
    if mode == 'train': # 60%
        images = images[:int(0.6*length)]
        labels = labels[:int(0.6*length)]
    elif mode == 'val': # 20%
        images = images[int(0.6 * length):int(0.8*length)]
        labels = labels[int(0.6 * length):int(0.8*length)]
    else:
        images = images[int(0.8 * length):]
        labels = labels[int(0.8 * length):]

    return images, labels, name2label

# imagenet 通用参数
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])
def normalize(x):
    # x: [224, 224, 3]
    return (x-mean)/std # 这里x 和 mean std 维度不同，是通过自动 broadcast 实现的计算

def denormalize(x):
    return x*std+mean

def process(x, y):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [244, 244])

    # 随机左右翻转
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_crop(x, [224, 224, 3]) # 随机裁剪部分

    # x [0~255] -> [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)

    y = tf.convert_to_tensor(y)

    return x, y

def main():
    images, labels, table = load_pokemon('./pokemon', 'train')
    print('images', len(images), images)
    print('labels', len(labels), labels)

    db = tf.data.Dataset.from_tensor_slices((images, labels))
    db = db.shuffle(1000).map(process).batch(32)

    print(db)


if __name__ == '__main__':
    main()
