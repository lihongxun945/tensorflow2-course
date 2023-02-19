import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras

from PIL import Image
import glob
from my_wgan import Generator, Discriminator

from dataset import make_anime_dataset

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)

def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]

    # [b, h, w, c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, True)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)

    return gp

def d_loss_fn(generator, discriminator, batch_x, batch_z, training=None):
    fake_images = generator(batch_z, training)

    fake_logits = discriminator(fake_images, training)
    real_logits = discriminator(batch_x, training)

    # 为了让结果正确，显然假图片应该判定为0， 真图片应该判定为1
    # 那么假图片的判定结果算一下平均值，应该尽量接近0，那么计算平均值就可以当做loss，平均值在0~1之间，显然越小越好
    fake_loss = tf.reduce_mean(fake_logits)
    # 同理，那么真图片平均值应该尽量接近1，越大越好，显然他的loss可以直接取负，这样就变成平均值越小越好
    real_loss = - tf.reduce_mean(real_logits)
    gp = gradient_penalty(discriminator, batch_x, fake_images)

    # 最后返回二者之和，让总共的平均值越小越好
    loss = fake_loss + real_loss + 10.*gp
    return loss, gp

def g_loss_fn(generator, discriminator, batch_z, training=None):
    fake_images = generator(batch_z, training)
    fake_logits = discriminator(fake_images, training)
    # 为了骗过判别器，显然应该是结果尽量接近1， 那么直接取反就能作为loss了
    return -tf.reduce_mean(fake_logits)


def main():
    tf.random.set_seed(233)
    np.random.seed(233)
    assert tf.__version__.startswith('2.')

    # hyper parameters
    z_dim = 100
    epochs = 3000000
    batch_size = 512
    learning_rate = 0.0005

    img_path = glob.glob(r'./faces/*.jpg')
    assert len(img_path) > 0
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = None
    USE_SAVED = True
    if (USE_SAVED and os.path.exists('./model.tf')):
        print('加载模型，继续上次训练')
        generator = tf.keras.models.load_model('./model.tf', compile=True)
    else:
        print('未找到保存的模型，重新开始训练')
        generator = Generator()
        generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        # 判别器每ge epoch进行5次迭代
        for _ in range(5):
            batch_z = tf.random.normal([batch_size, z_dim]) # 生成假图片
            batch_x = next(db_iter) # 加载真图片
            with tf.GradientTape() as tape:
                d_loss, gp = d_loss_fn(generator, discriminator, batch_x, batch_z, True)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        batch_z = tf.random.normal([batch_size, z_dim])  # 生成假图片
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, True)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        if epoch % 100 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), 'gp-loss:', float(gp))

            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('images', 'wgan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')
            # 子类实现的网络保存成h5格式不支持 参见 https://github.com/tensorflow/tensorflow/issues/29545
            generator.predict(z)  # 不调用一下，直接save 会报错，参见这里 https://github.com/tensorflow/tensorflow/issues/31057
            generator.save('./model.tf', overwrite=True, save_format="tf")


if __name__ == '__main__':
    main()
