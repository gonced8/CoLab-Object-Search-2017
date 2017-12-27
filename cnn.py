from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import skimage.transform
import numpy as np



def get_features (x):

    model = VGG16(weights='imagenet', include_top=False)

    # model.summary()

    if x.shape[1]!=512 or x.shape[2]!=512:
        x_resized = np.zeros((x.shape[0], 512, 512, x.shape[3]))

        for i in range (x.shape[0]):
            x_resized[i, :, :, :] = skimage.transform.resize(x[i, :, :, :], (512, 512), order=1, mode='reflect')

        x = x_resized

    x_pp = preprocess_input(x)

    features = model.predict(x_pp)

    return (x, x_pp, features)  # return preprocessed input data and return output
