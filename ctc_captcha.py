import tensorflow as tf
import captcha
import keras.backend as K
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, MaxPool2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from captcha.image import ImageCaptcha
import random
import matplotlib.pyplot as plt

import numpy as np
import string


class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 80
        self.img_cols = 170
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 36  # 待修改
        self.latent_dim = 128  # 待修改
        self.nb_character = 4
        self.classe_model_dir = 'ctc_loss.h5'
        self.cnn_w_dir = None
        self.characters = string.digits + string.ascii_uppercase

        optimizer = Adam(0.0002, 0.5)  # ？？？
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.D_regression = self.build_discriminator()
        self.D_regression.compile(loss=losses,
                                  optimizer=optimizer,
                                  metrics=['accuracy'])

        # Build class discriminator
        self.D_class = self.build_classes_discriminator()
        ## fixed model parameter
        self.D_class.trainable = False
        self.D_class.compile(loss='categorical_crossentropy',
                             optimizer='SGD',
                             metrics=['accuracy'])
        self.D_class.load_weights(self.cnn_w_dir)

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(4,))
        img = self.generator([noise, label])  # ＴＯＤＯ

        # For the combined model we will only train the generator
        self.D_regression.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        target_label = self.D_regression(img)  # TODO
        classes_label = self.D_class(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [target_label, classes_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)

    def real_data_gen(self,batch_size):
        X = np.zeros((batch_size, self.img_rows, self.img_cols, 3), dtype=np.float)
        y = [np.zeros((batch_size, self.num_classes), dtype=np.uint8) for i in range(self.nb_character)]
        generator = ImageCaptcha(width=self.img_cols, height=self.img_rows)
        while True:
            for i in range(batch_size):
                random_str = ''.join([random.choice(self.characters) for j in range(4)])
                # x_t = generator.generate_image(random_str) TODO
                # X[i] = (x_t - 127.5) / 127.5
                X[i] = generator.generate_image(random_str)
                for j, ch in enumerate(random_str):
                    y[j][i, :] = 0
                    y[j][i, self.characters.find(ch)] = 1
            yield X, y

    def build_classes_discriminator(self):
        img = Input((self.img_shape))
        x = img
        for i in range(3):
            x = Conv2D(32 * 2 ** i, 3, activation='relu')(x)
            x = Conv2D(32 * 2 ** i, 3, activation='relu')(x)
            x = MaxPool2D((2, 2))(x)

        x = Flatten()(x)
        x = Dropout(0.25)(x)
        x = [Dense(62, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
        model = Model(img, x)

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
        label = Input(shape=(4,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))  # 4*128
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
            print(x.get_shape().as_list())

            if i == 1 or i == 3:
                print(i)
                x = ZeroPadding2D(padding=(0, 1))(x)
            x = Conv2D(f, k, padding='same')(x)
            # shape = x.get_shape().as_list()
            # print(shape)
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

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, 10, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
            img_labels = y_train[idx]
            fake_labels = 10 * np.ones(img_labels.shape)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=14000, batch_size=32, sample_interval=200)
