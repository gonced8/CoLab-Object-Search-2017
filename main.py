# This is the main function file.
# The code is a bit confuse because it was designed to process all the images at once. But due to memory insufficiency it was modified to process an image at a time


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
from plot_sequence import plot_sequence_on_image


# This function is used to calculate the probability of a sequence using the trained RNN
def calc_prob(seqs, size, empty):
    new_seqs = [np.pad(seq, (0, size - seq.size), 'constant', constant_values=empty) for seq in seqs]
    new_feat_seqs = [ind_to_features(x_test_features_reshaped, seq, empty) for seq in new_seqs]

    probs = model_rnn.predict(np.array(new_feat_seqs))

    return probs.reshape(probs.shape[0])


# This is an auxiliar function to get the features from an index
def ind_to_features(features, ind_seq, empty):
    return [(features[int(i)] if i != empty else np.zeros(features.shape[1])) for i in ind_seq]



# The interval of images to process
start=0
end=12500

count=end-start
j=int(count/100)
print("Data / Features / Saliency")
#load and pre-process images
pre_process=False
generate_sequences=False
for i in (range(count)):

    # Printing the progress
    
    
    if pre_process:
    
      if i%j==0:
        print(str(int(100.*i/count))+" of 100%")
#    print(i)

      # Getting the training data
  #    print("Data")
      (x_train, _) = data.get_train_dogs_vs_cats(0, start+i, start+i+1)  # gets the train data (only cats) from dataset dogs vs cats
  
      # Extracting the training features
  #    print("Features")
      (_, _, x_features) = cnn.get_features(x_train)  # gets resized data, preprocessed data and corresponding features (output of CNN)
      keras.backend.clear_session()
  
      # Calculating the training saliency
  #    print("Saliency")
      (_, x_saliency) = saliency.get_saliency(x_train, 16)  # gets saliency map of 16x16
      keras.backend.clear_session()
  
      # Saving the image and the corresponding features and saliency in a numpy file
      np.savez_compressed("outfile/out"+str(i+start), x_train=x_train[0], x_features=x_features[0], x_saliency=x_saliency[0])
    elif generate_sequences:
      # Loading the image, features and saliency to then calculate the training sequences
      data=np.load("outfile/out"+str(i+start)+".npz")
  
      # Sequences generation
  #    print("Sequence")
      (x_sequence, x_sequence_index, y) = sequence.generate_sequence(np.expand_dims(data["x_features"], axis=0), np.expand_dims(data["x_saliency"], axis=0), 10, 2, 2)  # gets 2 correct and 2 wrong sequences of size 10 for each image
  
      # Saving the sequences
  #    print("Save")
      np.savez_compressed("sequence/seq"+str(i), x_sequence=x_sequence, x_sequence_index=x_sequence_index, y=y)
  
      keras.backend.clear_session()



# RNN training

train=False

if train:
	data=np.load("sequence/seq0.npz")
	x_sequence=np.empty((0, data['x_sequence'].shape[1], data['x_sequence'].shape[2], data['x_sequence'].shape[3]), dtype=data['x_sequence'].dtype)
	y=np.empty((0, data['y'].shape[1]), dtype=data['y'].dtype)

	model_rnn = rnn.create(x_sequence.shape[2])

	# Interval of images (sequences) to use for training
	start=0
	end=12500
	batch=500

	count=end-start
	j=int(count/100)
	print("Load")
	for i in (range(count)):

		if i%j==0:
		    print('\n'+str(int(100.*i/count))+" of 100%")

		# Getting the sequences to use for training
		data=np.load("sequence/seq"+str(i+start)+".npz")
		x_sequence=np.append(x_sequence, data['x_sequence'], axis=0)
		y=np.append(y, data['y'], axis=0)

		x_sequence_flat = x_sequence.reshape(x_sequence.shape[0] * x_sequence.shape[1], x_sequence.shape[2], x_sequence.shape[3])

		# If the batch size is correct, then train the RNN
		if (i+1)%batch==0:
		    #print("Fit")
		    rnn.fit(model_rnn, x_sequence_flat, y.flatten())
		
		    # Erasing the batch to then get the new one
		    del(x_sequence)
		    del(y)
		    del(x_sequence_flat)
		    x_sequence=np.empty((0, data['x_sequence'].shape[1], data['x_sequence'].shape[2], data['x_sequence'].shape[3]), dtype=data['x_sequence'].dtype)
		    y=np.empty((0, data['y'].shape[1]), dtype=data['y'].dtype)


test=False

if test:

	# Testing of the RNN

	model_rnn = keras.models.load_model('model.h5')

	# Uses the test images to test
	#(x_test, _) = data.get_test_dogs_vs_cats(10)

	# Uses the train images to test
	(x_test, _) = data.get_train_dogs_vs_cats(0, 0, 10)

	(_, _, x_test_features) = cnn.get_features(x_test)


	x_test_sequence = np.empty((0, 10), dtype=int)

	for i in tqdm(range(x_test.shape[0])):
		x_test_features_reshaped = x_test_features[i].reshape(x_test_features.shape[1] * x_test_features.shape[2], x_test_features.shape[3])
		seq = beam_search.search(calc_prob, 10, range(256), 5)
		x_test_sequence = np.append(x_test_sequence, [seq], axis=0)
		np.savez_compressed("npseq2/seq"+str(i), img=x_test[i], seq=seq)

		sys.stdout.flush()

	sys.exit()


# This function is used to plot the generated testing sequence
# The input is the index of the image
plot_sequence_on_image(4, 1)

#######################################################################################################################################################


# beam_search.search(calc_prob, 3, range(256), 3)
# TEST:  model_rnn.predict(np.array([ind_to_features(x_test_features_reshaped, [ 120.,  101.,  121.], float('inf'))]))




