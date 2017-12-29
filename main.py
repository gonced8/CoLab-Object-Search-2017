import keras
import numpy as np

import beam_search
import data
import cnn
import rnn
import saliency
import sequence


# (x_train, _) = data.get_train_cifar10() # gets the train data from dataset cifar10

(x_train, _) = data.get_train_dogs_vs_cats(0, 3)  # gets the train data (only cats) from dataset dogs vs cats

(x_train_r, _, x_features) = cnn.get_features(x_train)  # gets resized data, preprocessed data and corresponding features (output of CNN)

(_, x_saliency) = saliency.get_saliency(x_train, 16)  # gets saliency map of 16x16

(x_sequence, x_sequence_index, y) = sequence.generate_sequence(x_features, x_saliency, 3, 2, 3)  # gets 2 correct and 4 wrong sequences of size 3 for each image

x_sequence_flat = x_sequence.reshape(x_sequence.shape[0] * x_sequence.shape[1], x_sequence.shape[2],
                                     x_sequence.shape[3])

model_rnn = rnn.create(x_sequence_flat.shape[1])

rnn.fit(model_rnn, x_sequence_flat, y.flatten())

(x_test, _) = data.get_train_dogs_vs_cats(0, 4)
(_, _, x_test_features) = cnn.get_features(np.array([x_test[-1]]))

x_test_features_reshaped = x_test_features[0].reshape(x_test_features.shape[1] * x_test_features.shape[2], x_test_features.shape[3])


# seq = beam_search.search(model_rnn, x_test_features_reshaped)


def calc_prob(seqs, size, empty):
    new_seqs = [np.pad(seq, (0, size - seq.size), 'constant', constant_values=empty) for seq in seqs]
    new_feat_seqs = [ind_to_features(x_test_features_reshaped, seq, empty) for seq in new_seqs]

    probs = model_rnn.predict(np.array(new_feat_seqs))

    return probs.reshape(probs.shape[0])


def ind_to_features(features, ind_seq, empty):
    return [(features[int(i)] if i != empty else np.zeros(features.shape[1])) for i in ind_seq]


# beam_search.search(calc_prob, 3, range(256), 3, 0)
# TEST:  model_rnn.predict(np.array([ind_to_features(x_test_features_reshaped, [ 120.,  101.,  121.], float('inf'))]))
