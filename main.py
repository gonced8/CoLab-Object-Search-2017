import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import data
import cnn
import rnn
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

model = VGG16(weights='imagenet', include_top=True)
model.summary()

(x_train, y_train, x_test, y_test) = data.get_train()



# (x_train_p, _, features) = cnn.get_features(x_train)    # gets preprocessed data and corresponding features (output of cnn)

#class_idx = 0
#indices = np.where(y_test[:, 0] == class_idx)[:, 0]

#len(indices)

# pick some random input from here.
idx = 0

# Lets sanity check the picked image.

plt.figure(1)
plt.rcParams['figure.figsize'] = (18, 6)

plt.imshow(x_test[idx][...])

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

grads = visualize_saliency(model, layer_idx, None, seed_input=x_test[0:1, :, :, :], backprop_modifier='guided')

print('d')
# Plot with 'jet' colormap to visualize as a heatmap.
plt.figure(2)
plt.imshow(grads, cmap='jet')


# model_rnn = rnn.create()
