from keras import layers
from keras import models
import numpy as np


def create():

    model_rnn = models.Sequential()
    model_rnn.add(layers.LSTM(64, input_shape=(20, 512)))    #15000 sequences and (example) 10 steps per sequence
    model_rnn.add(layers.Dropout(0.2))
    model_rnn.add(layers.Dense(1, activation='sigmoid'))
    model_rnn.compile(loss='mean_squared_error', optimizer='rmsprop')

    model_rnn.summary()

    return model_rnn


def fit(model_rnn, x, y):

    model_rnn.fit(x, y), epochs=20, batch_size=64)

    filepath = 'weights.h5'

    model_rnn.save_weights(filepath)


    #filepath="weights-improvement.h5"
    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    #callbacks_list = [checkpoint]
    #model_rnn.fit(x, y, epochs=20, batch_size=64, callbacks=callbacks_list)
