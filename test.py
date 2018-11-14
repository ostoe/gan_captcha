from keras.models import *
from keras.layers import *
import keras
input_tensor = Input((64, 64, 3))
x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, 3, activation='relu', padding='same')(x)
    x = Convolution2D(32*2**i, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    print(x.shape)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(10, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(input_tensor, x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


def build_cnn_plus():
    input_tensor = Input((64, 64, 3))
    x = input_tensor
    # conv1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(2, 2, 'same')(x)
    # conv2
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(2, 2, 'same')(x)
    # conv3
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(2, 2, 'same')(x)
    # conv4
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(2, 2, 'same')(x)
    # fc1
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.5)(x)
    # fc2
    x = Dense(10 * 4)(x)

    x = Reshape([4, 10])(x)
    x = Softmax()(x)

    opt = keras.optimizers.Adam(1e-4)
    m = Model(input_tensor, x)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
import numpy as np
X = np.random.randn(32,64,64,3)
y = [np.random.randn(32,10) for _ in range(10)]

model1 = build_cnn_plus()

