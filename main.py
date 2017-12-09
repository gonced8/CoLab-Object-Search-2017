import numpy as np
import data
import cnn
import rnn

(x_train, _) = data.get_train() #

(x_train_r, _, features) = cnn.get_features (x_train)    # gets resized data, preprocessed data and corresponding features (output of CNN)

model_rnn = rnn.create()
