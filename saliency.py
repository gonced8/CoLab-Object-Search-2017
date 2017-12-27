import numpy as np
import skimage.transform
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
import data
from keras import layers
from keras import models



def calculate_saliency(x):

    model = VGG16(weights='imagenet', include_top=True)
    # model.summary()

    # (x, _) = data.get_train_dogs_vs_cats(0, 5) # gets the train data (only cats) from dataset dogs vs cats

    msize=224

    if x.shape[1]!=msize or x.shape[2]!=msize:
        x_resized = np.zeros((x.shape[0], msize, msize, x.shape[3]))

        for i in range (x.shape[0]):
            x_resized[i, :, :, :] = skimage.transform.resize(x[i, :, :, :], (msize, msize), order=0, mode='reflect')

        x = x_resized

    # x.shape
    # pick some random input from here.
    # idx = 2

    # Lets sanity check the picked image.

    # plt.figure(1)
    # plt.rcParams['figure.figsize'] = (18, 6)
    # plt.imshow(x[idx]/255.)
    # plt.show(block=False)

    #print('a')

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'predictions')

    #print('b')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    #print('c')
    #print(x_test.shape)
    #x_test = np.swapaxes(x_test, 1, 2)
    #x_test = np.swapaxes(x_test, 2, 3)
    #print(x_test.shape)
    #x.shape

    x_pp = preprocess_input(x)

    # x_pp.shape

    sal = np.empty((0, x_pp.shape[1], x_pp.shape[2], 1), dtype=int)

    for i in range (x_pp.shape[0]):

        sal = np.append(sal, [visualize_saliency(model, layer_idx, None, seed_input=x_pp[i], backprop_modifier='guided')[:, :, 2:3]], axis=0)   # only the last rgb channel contains information

    # sal.shape

    #print('d')

    # Plot with 'jet' colormap to visualize as a heatmap.
    # plt.figure(2)
    # plt.imshow(sal, cmap='jet')
    # plt.show(block=False)

    return sal



def pooling (x, n):

    x.shape
    model = models.Sequential()
    model.add(layers.AveragePooling2D(pool_size=(int(x.shape[1]/n), int(x.shape[2]/n)), strides=int(x.shape[1]/n), input_shape=(x.shape[1], x.shape[2], x.shape[3])))

    output = model.predict(x)
    output = np.squeeze(output, axis=-1)

    return output



def get_saliency (x, n):

    sal = calculate_saliency(x)

    sal16 = pooling (sal, 16)

    return (sal, sal16)
