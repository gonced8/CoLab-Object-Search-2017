import numpy as np
import data
import cnn
import rnn

(x_train, _) = data.get_train_cifar10() # gets the train data from dataset cifar10

(x_train, _) = data.get_train_dogs_vs_cats(0, 1000) # gets the train data (only cats) from dataset dogs vs cats

(x_train_r, _, features) = cnn.get_features (x_train)    # gets resized data, preprocessed data and corresponding features (output of CNN)

model_rnn = rnn.create()
