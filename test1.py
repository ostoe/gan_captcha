import time
import cv2

import numpy as np
import matplotlib.pyplot as plt

img = np.random.randint(0,255, size=(100,100,3))
dir_path = 'ffffffsf/'
import glob
img_list = glob.glob('.jpg')
for x in img_list:
    imgs = cv2.imread()
plt.imshow(img)
plt.show()

s1 = time.time()
time.sleep(0.5)
s2 = time.time()
print(s2-s1)
