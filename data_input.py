from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string


ImageCaptcha()
class DATA(object):
    def __init__(self, shape, length):
        self.size = shape
        self.length = length
        self.generator = ImageCaptcha(width=self.size[1], height=self.size[0])
        self.char = string.digits + string.ascii_lowercase + string.ascii_uppercase
        self.nb_len = len(self.char)

    def gen(self, batch_size=32):
        X = np.zeros((batch_size, self.size[0], self.size[1], 3), dtype=np.uint8)
        y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
        generator = ImageCaptcha(width=width, height=height)
        while True:
            for i in range(batch_size):
                random_str = ''.join([random.choice(characters) for j in range(4)])
                X[i] = generator.generate_image(random_str)
                for j, ch in enumerate(random_str):
                    y[j][i, :] = 0
                    y[j][i, characters.find(ch)] = 1
            yield X, y
