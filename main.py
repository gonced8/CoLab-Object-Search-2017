import numpy as np
import cnn
import data

(x_train, y_train) = data.get_train()

(x_train_p, _, features) = cnn.get_features (x_train)    # gets preprocessed data and corresponding features (output of cnn)
