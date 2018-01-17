import keras
import numpy as np
import sys
from tqdm import tqdm

import beam_search
import cnn
import data
import rnn
import saliency
import sequence


def calc_prob(seqs, size, empty):
    new_seqs = [np.pad(seq, (0, size - seq.size), 'constant', constant_values=empty) for seq in seqs]
    new_feat_seqs = [ind_to_features(x_test_features_reshaped, seq, empty) for seq in new_seqs]

    probs = model_rnn.predict(np.array(new_feat_seqs))

    return probs.reshape(probs.shape[0])


def ind_to_features(features, ind_seq, empty):
    return [(features[int(i)] if i != empty else np.zeros(features.shape[1])) for i in ind_seq]



count=1

for i in range(count):

    print("Data")
    (x_train, _) = data.get_train_dogs_vs_cats(0, i+1, True)  # gets the train data (only cats) from dataset dogs vs cats

    print("Features")
    (_, _, x_features) = cnn.get_features(x_train)  # gets resized data, preprocessed data and corresponding features (output of CNN)

    print("Saliency")
    (_, x_saliency) = saliency.get_saliency(x_train, 16)  # gets saliency map of 16x16

    np.savez_compressed("outfile"+str(i), x_train=x_train, x_features=x_features, x_saliency=x_saliency)

sys.exit()

print("Sequence")
(x_sequence, x_sequence_index, y) = sequence.generate_sequence(x_features, x_saliency, 10, 2, 2)  # gets 2 correct and 4 wrong sequences of size 3 for each image

x_sequence_flat = x_sequence.reshape(x_sequence.shape[0] * x_sequence.shape[1], x_sequence.shape[2], x_sequence.shape[3])

model_rnn = rnn.create(x_sequence_flat.shape[1])

print("Fit")
rnn.fit(model_rnn, x_sequence_flat, y.flatten())


(x_test, _) = data.get_test_dogs_vs_cats(0, 10)
(x_test_r, _, x_test_features) = cnn.get_features(x_test)

file = open("train_sequences.txt", "w")
x_test_sequence = np.empty((0, 10), dtype=int)

for i in tqdm(range(x_test.shape[0])):
    x_test_features_reshaped = x_test_features[i].reshape(x_test_features.shape[1] * x_test_features.shape[2], x_test_features.shape[3])
    seq = beam_search.search(calc_prob, 10, range(256), 5)
    x_test_sequence = np.append(x_test_sequence, [seq], axis=0)
    file.write("%s\n" % seq)

file.close()


#######################################################################################################################################################


# beam_search.search(calc_prob, 3, range(256), 3)
# TEST:  model_rnn.predict(np.array([ind_to_features(x_test_features_reshaped, [ 120.,  101.,  121.], float('inf'))]))
