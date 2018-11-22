import tensorflow as tf
import captcha
import keras.backend as K
# from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, MaxPool2D, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from captcha.image import ImageCaptcha
import random
import matplotlib.pyplot as plt
# import tensorflow.keras.backend  as K
import numpy as np
import string, os


class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 80
        self.img_cols = 170
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 36  # 待修改
        self.latent_dim = 128  # 待修改
        self.length_char = 4
        self.cnn_loss_weights = 'cnn_4_w.h5'
        self.saved_model_dir = 'saved_model'
        self.saved_images_dir = 'cpatcha_images'
        self.characters = string.digits + string.ascii_uppercase

        D_opt = Adam(0.0001, 0.0, 0.9)  # ？？？
        G_opt = Adam(0.0001, 0.0, 0.9)
        losses = ['binary_crossentropy'] + 4 * ['sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.D_regression = self.build_d_regression()
        self.D_regression.compile(loss='binary_crossentropy',
                                  optimizer=G_opt,
                                  metrics=['accuracy'])

        # Build class discriminator
        self.D_class = self.build_classes_discriminator()
        ## fixed model parameter
        self.D_class.trainable = False
        self.D_class.compile(loss='sparse_categorical_crossentropy',
                             optimizer='SGD',
                             metrics=['accuracy'])
        self.D_class.load_weights(self.cnn_loss_weights)  #

        # Build the generator
        self.generator = self.build_generator_custom()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.length_char,))
        img = self.generator([noise, label])  # TODO
        img = Lambda(lambda x: x * 127.5 + 127.5)(img)  # 变换

        # For the combined model we will only train the generator
        self.D_regression.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        target_label = self.D_regression(img)  # TODO
        classes_label = self.D_class(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [target_label] + classes_label)  # [1/0, a,b,c,d]
        self.combined.compile(loss=losses,
                              optimizer=G_opt,
                              metrics=['accuracy'],
                              loss_weights=[1., 0.25, 0.25, 0.25, 0.25])

    def real_data_gen(self, batch_size):
        X = np.zeros((batch_size, self.img_rows, self.img_cols, 3), dtype=np.float)
        y = [np.zeros((batch_size, self.num_classes), dtype=np.uint8) for i in range(self.length_char)]
        generator = ImageCaptcha(width=self.img_cols, height=self.img_rows)
        while True:
            for i in range(batch_size):
                random_str = ''.join([random.choice(self.characters) for j in range(4)])
                x_t = generator.generate_image(random_str)  # TODO
                X[i] = np.asarray(x_t) / 255.0
                for j, ch in enumerate(random_str):
                    y[j][i, :] = 0
                    y[j][i, self.characters.find(ch)] = 1
            yield X, y

    def build_classes_discriminator(self):  # TODO same to ipy
        img = Input((self.img_shape))
        x = img
        # for i in range(4):
        #     x = Conv2D(32 * 2 ** i, 3, padding='same')(x)
        #     x = BatchNormalization()(x)
        #     x = Activation('relu')(x)
        #
        #     x = Conv2D(32 * 2 ** i, 3, padding='same')(x)
        #     x = BatchNormalization()(x)
        #     x = Activation('relu')(x)
        #
        #     x = MaxPool2D((2, 2))(x)
        for i in range(4):
            x = Conv2D(32 * 2 ** i, 3, activation='relu')(x)
            x = Conv2D(32 * 2 ** i, 3, activation='relu')(x)
            x = MaxPool2D((2, 2))(x)

        x = Flatten()(x)
        x = Dropout(0.0)(x)
        x = [Dense(self.num_classes, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
        model = Model(img, x)

        return model

    def build_d_regression(self):  # TODO
        input_layers = Input((self.img_shape))
        x = input_layers
        for i in range(3):
            x = Conv2D(32 * 2 ** i, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if i > 1:
                x = MaxPool2D((2, 2))(x)  # add two pooling
            x = Conv2D(32 * 2 ** i, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = MaxPool2D((2, 2))(x)
        x = Flatten()(x)
        output = Dense(1, activation="sigmoid")(x)
        print(output)

        model = Model(input_layers, output)
        return model

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(4,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_generator_custom(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.length_char,), dtype='int32')
        label_embedding = (Embedding(self.num_classes, self.latent_dim)(label))  # 36*128
        x = multiply([noise, label_embedding])
        x = Flatten()(x)

        x = Dense(128 * 5 * 10)(x)
        x = Reshape((5, 10, 128))(x)
        filters_size = [256, 128, 64, 32]
        kernel_size = [5, 5, 5, 5]
        index = list(range(len(filters_size)))

        for i, f, k, in zip(index, filters_size, kernel_size):
            #   x = tf.layers.UpSampling2D()
            x = BatchNormalization()(x)
            x = UpSampling2D(interpolation='bilinear')(x)
            #     print(x.get_shape().as_list())

            x = Conv2D(f, k, padding='same')(x)
            x = Activation("relu")(x)
            # print(x.get_shape().as_list())

            if i == 1 or i == 3:
                print(i)
                x = ZeroPadding2D(padding=(0, 1))(x)
            x = Conv2D(f, k, padding='same')(x)
            # shape =
            print(x.get_shape().as_list())
        x = Conv2D(3, 5, padding='same')(x)
        output = Activation("tanh")(x)

        model = Model([noise, label], output)
        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes + 1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=32, g_train_interval=5, sample_interval=50, use_pretrain=False):
        # use pretrain
        if use_pretrain:
            self.generator.load_weights(os.path.join(self.saved_model_dir, 'generator_weights.h5'))
            self.D_class.load_weights(os.path.join(self.saved_model_dir, 'D_class_weights.h5'))
            self.D_regression.load_weights(os.path.join(self.saved_model_dir, 'D_regression_weights.h5'))
        generator = ImageCaptcha(width=self.img_cols, height=self.img_rows)
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            y_char_label = np.zeros((batch_size, self.length_char), dtype=np.uint8)

            real_images = np.zeros((batch_size, self.img_rows, self.img_cols, self.channels), dtype=np.uint8)

            for i in range(batch_size):
                random_str = ''.join([random.choice(self.characters) for j in range(self.length_char)])
                real_images[i] = generator.generate_image(random_str)  # TODO

                y_tmp = [self.characters.find(x) for x in random_str]
                y_char_label[i] = y_tmp

            ######  UNIT TEST ######
            # def unit_test(img, label):
            #     s = ''.join([self.characters[x] for x in label])
            #     print('label',s)
            #     plt.imshow(img)
            #     plt.show()
            # unit_test(real_images[0], y_char_label[0])
            # print(real_images.shape, y_char_label.shape)
            ##########      #####

            # sample captcha_images form generator
            # confus
            if random.random() > 0.5:
                confus_label = y_char_label
            else:
                confus_label = np.random.randint(0, self.num_classes, (batch_size, self.length_char))
            sample_images = self.generator.predict([noise, confus_label])
            g_images = (lambda x: x * 127.5 + 127.5)(sample_images)
            d_real_metrics = self.D_regression.train_on_batch(real_images, real_labels)
            d_fake_metrics = self.D_regression.train_on_batch(g_images, fake_labels)
            d_metrics = 0.5 * np.add(d_real_metrics, d_fake_metrics)

            #
            #
            # # Select a random batch of captcha_images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]
            #
            # # Sample noise as generator input
            # noise = np.random.normal(0, 1, (batch_size, 100))
            #
            # # The labels of the digits that the generator tries to create an
            # # image representation of
            # sampled_labels = np.random.randint(0, 10, (batch_size, 1))
            #
            # # Generate a half batch of new captcha_images
            # gen_imgs = self.generator.predict([noise, sampled_labels])
            #
            # # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
            # img_labels = y_train[idx]
            # fake_labels = 10 * np.ones(img_labels.shape)
            # # 原作里面的g入口的条件是随机的，和real入口的image生成不是一个条件
            # # Train the discriminator
            # d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            #
            #

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            for _ in range(g_train_interval):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                y_char_label = np.random.randint(0, self.num_classes, (batch_size, self.length_char))
                y1, y2, y3, y4 = y_char_label[:, 0], y_char_label[:, 1], y_char_label[:, 2], y_char_label[:, 3]
                # print(y_char_label[0], y1[0], y2[0], y3[0], y4[0])
                g_metrics = self.combined.train_on_batch([noise, y_char_label], [real_labels, y1, y2, y3, y4])

                # g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G lb-loss: %f, class-loss: %f, D_acc:%.4f%% acc: %.4f%%]" % (
                epoch, d_metrics[0],
                100 * d_metrics[1],
                g_metrics[1],
                np.mean(g_metrics[2:6]),
                100. * g_metrics[6],
                100 * np.mean(g_metrics[7:11])
            ))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # self.save_model()
                self.sample_images(epoch)
            if epoch % sample_interval * 5 == 0:
                self.save_model()

    def sample_images(self, epoch):
        r, c = 3, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.random.randint(0, self.num_classes, (r * c, self.length_char))
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale captcha_images 0 - 1
        gen_imgs = 127.5 * gen_imgs + 127.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("captcha_images/1%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s1.json" % model_name
            weights_path = "saved_model/%s_weights1.h5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.D_regression, "D_regression")
        save(self.D_class, 'D_class')


if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=180000, batch_size=128, g_train_interval=10, sample_interval=100, use_pretrain=False)
