'''
Functions to get the training and testing data
'''

import numpy as np
from keras.datasets import cifar10
from keras.preprocessing import image
import os
from tqdm import tqdm


# This function returns the cifar10 data. This is obselete since was used only during the first tests
def get_train_cifar10():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_test.shape
    y_test.shape

    chosen_class=0
    x = np.empty((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]), dtype=int)
    y = np.empty((0, 1), dtype=int)

    x_train[0].shape
    for i in range (y_test.shape[0]):
        if y_test[i, 0]==chosen_class:
            x = np.append(x, [x_train[i]], axis=0)
            y = np.append(y, [y_train[i]], axis=0)

    return (x, y)


# This functions returns the dogs vs cats training data. Use choose to select what type of images (0=cat, 1=dog, 2=both). You can select an image interval to get with start and end.
def get_train_dogs_vs_cats(choose=2, start=0, end=-1):

    count=0
    
    # Path to the folder with the training data
    path = './data/dogs vs cats/train'
    
    # x will contain the images and y their classification (dog or cat)
    x = np.empty((0, 512, 512, 3), dtype=int)
    y = np.empty((0, 1), dtype=int)

    if end==-1:
        end=len(list(os.listdir(path)))

    for name in (os.listdir(path)):
    #for name in tqdm(os.listdir(path)):

        if name[0:3]=='cat' and choose==1:
            continue;
        if name[0:3]=='dog' and choose==0:
            continue

        if count>=start and count<end: 
            img_path = os.path.join(path, name)
            img = image.load_img(img_path, target_size=(512, 512))
            array = image.img_to_array(img)
            array.shape
            x = np.append(x, [array], axis=0)
            y = np.append(y, [[int(name[0:3]=='dog')]], axis=0)

        elif count>=end:
            break

        count+=1

    return (x, y)


# This function returns the dogs vs cats testing data. Use number to select the number of images you want to return
def get_test_dogs_vs_cats(number=-1):

    count=0
    
    # Path to the folder with the training data
    path = './data/dogs vs cats/test'

    x = np.empty((0, 512, 512, 3), dtype=int)

    for name in (os.listdir(path)):
    #for name in tqdm(os.listdir(path)):

        if count==number:
            break
        else:
            count+=1

        img_path = os.path.join(path, name)
        img = image.load_img(img_path, target_size=(512, 512))
        array = image.img_to_array(img)

        x = np.append(x, [array], axis=0)

    return (x, y)
