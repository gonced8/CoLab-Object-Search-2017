import numpy as np
from keras import layers
from keras import models
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


model = VGG16(weights='imagenet', include_top=False)
