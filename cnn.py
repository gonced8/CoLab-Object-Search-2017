'''
This function receives an array of name x and returns its features
'''


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import skimage.transform
import numpy as np
from tqdm import tqdm

def get_features (x):

    model = VGG16(weights='imagenet', include_top=False)

    # model.summary()

    # Resizing of the image to 512x512 so that the cnn, after the pooling, returns 512 features
    if x.shape[1]!=512 or x.shape[2]!=512:
        x_resized = np.zeros((x.shape[0], 512, 512, x.shape[3]))

        for i in (range (x.shape[0])):
        #for i in tqdm(range (x.shape[0])):
            x_resized[i, :, :, :] = skimage.transform.resize(x[i, :, :, :], (512, 512), order=1, mode='reflect')

        x = x_resized

    x_pp = preprocess_input(x)

    # Features extraction
    features = model.predict(x_pp)

    return (x, x_pp, features)  # return preprocessed input data and return output
