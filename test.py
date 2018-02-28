# This function was previously used to test the use of PIL library to show images

import skimage.transform
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.nan)

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image


pic = Image.open("data/dogs vs cats/train/cat.0.jpg")

x = np.array(pic)

x.shape
x[:,:, 0]

x2=skimage.transform.resize(x, (32, 32, 3), order=1, mode='reflect')

x2=(x2*255.1).astype('uint8')

x==x2

x2.shape

img2 = Image.fromarray(x2, 'RGB')
img2.show()

x2=x2.astype('float64')
x2 = preprocess_input(x2)
x2[:, : , 0]
x2=(x2).astype('uint8')

img3 = Image.fromarray(x2, 'RGB')
img3.show()
