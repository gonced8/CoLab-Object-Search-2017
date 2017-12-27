import numpy as np
import data
import cnn
import rnn
import saliency
import sequence


# (x_train, _) = data.get_train_cifar10() # gets the train data from dataset cifar10

(x_train, _) = data.get_train_dogs_vs_cats (0, 2) # gets the train data (only cats) from dataset dogs vs cats

(x_train_r, _, x_features) = cnn.get_features (x_train)    # gets resized data, preprocessed data and corresponding features (output of CNN)

(_, x_saliency) = saliency.get_saliency (x_train, 16)   # gets saliency map of 16x16

(x_sequence, x_sequence_index, y) = sequence.generate_sequence(x_features, x_saliency, 3, 2, 2) # gets 2 correct and 4 wrong sequences of size 3 for each image

# model_rnn = rnn.create()
