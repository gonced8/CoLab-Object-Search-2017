from keras.datasets import cifar10
import numpy as np


def get_train():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_test.shape
    y_test.shape

    chosen_class=0
    x = np.empty((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]), dtype=int)
    y = np.empty((0, 1), dtype=int)

    for i in range (y_test.shape[0]):
        if y_test[i, 0]==chosen_class:
            x = np.append(x, [x_train[i]], axis=0)
            y = np.append(y, [y_train[i]], axis=0)

    return (x, y)
