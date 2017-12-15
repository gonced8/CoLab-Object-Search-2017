import numpy as np
import skimage.transform
from keras.applications import VGG16
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

import data

def get_saliency(x, y)

    model = VGG16(weights='imagenet', include_top=True)
    model.summary()

    msize=224

    (x, _) = data.get_train_dogs_vs_cats(0, 5) # gets the train data (only cats) from dataset dogs vs cats


    if x.shape[1]!=msize or x.shape[2]!=msize:
        x_resized = np.zeros((x.shape[0], msize, msize, x.shape[3]))

        for i in range (x.shape[0]):
            x_resized[i, :, :, :] = skimage.transform.resize(x[i, :, :, :], (msize, msize), order=0, mode='reflect')

        x = x_resized

    x.shape
    # pick some random input from here.
    idx = 4

    # Lets sanity check the picked image.

    plt.figure(1)
    plt.rcParams['figure.figsize'] = (18, 6)

    plt.imshow(x[idx][...])

    print('a')

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'predictions')

    print('b')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    print('c')

    #print(x_test.shape)
    #x_test = np.swapaxes(x_test, 1, 2)
    #x_test = np.swapaxes(x_test, 2, 3)
    #print(x_test.shape)

    x.shape
    grads = visualize_saliency(model, layer_idx, None, seed_input=x[4:5, :, :, :], backprop_modifier='guided')

    print('d')
    # Plot with 'jet' colormap to visualize as a heatmap.
    plt.figure(2)
    plt.imshow(grads, cmap='jet')
