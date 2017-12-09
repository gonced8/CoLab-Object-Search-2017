import numpy as np
import data
import cnn
import rnn

(x_train, y_train) = data.get_train()

(x_train_p, _, features) = cnn.get_features (x_train)    # gets preprocessed data and corresponding features (output of cnn)

model_rnn = rnn.create()
