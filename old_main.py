from keras import layers
from keras import models
from keras.datasets import cifar10
import numpy as np
import auxfunctions



#Convolutional Neural Network

model_cnn = models.Sequential()
model_cnn.add(layers.UpSampling2D(size=(16, 16), input_shape=(32, 32, 3)))
model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
#model_cnn.add(layers.Conv2D(32, (3, 3), input_shape=(512, 512, 3), activation='relu', padding='same'))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))

model_cnn.summary()

#Extractring Features

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train.shape
x_test.shape
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

features = model_cnn.predict(x_test[:, : , : , :])
features.shape

sequence = np.zeros((x_test.shape[0], 20, 512))  #15000 sequences and 10 steps per sequence and 512 features
sequence.shape

for i in range(sequence.shape[0]):
    for j in range(sequence.shape[1]):
        sequence[i, j] = (features[i, np.random.randint(16) , np.random.randint(16), :])


#Training Recurrent Neural Network

    # feed to the RNN 256 sequences of the same image
    # 1 of the 256 will have the correct prediction, returning the value 1 for the correct and 0 for the others

model_rnn = models.Sequential()
model_rnn.add(layers.LSTM(64, input_shape=(20, 512)))    #15000 sequences and (example) 10 steps per sequence
model_rnn.add(layers.Dropout(0.2))
model_rnn.add(layers.Dense(1, activation='sigmoid'))
model_rnn.compile(loss='mean_squared_error', optimizer='rmsprop')

model_rnn.summary()

model_rnn.fit(sequence, np.ones(sequence.shape[0]), epochs=10, batch_size=64)


#Predicting with Recurrent Neural Network

    # generate a starting sequence
    # create 256 sequences by adding a different position in the end of the starting sequence
    # feed the sequences to the RNN and chose 20 with the higher probabilities (output)
    # repeat and stop when the desired size is achieved

random = np.random.randint(x_test.shape[0])
test_features = model_cnn.predict(x_test[random:random+1, : , : , :])

test_sequence = np.zeros((16*16, 20, 512))

position = np.zeros((16*16, 20))

starting_sequence = np.random.randint(256, size=19)

for i in range(test_sequence.shape[0]):
    for j in range(starting_sequence.shape[0]):
        test_sequence[i, j] = test_features[0, int(starting_sequence[j]/16), starting_sequence[j]%16, :]
        position[i, j] = starting_sequence[j]

    test_sequence[i, -1] = test_features[0, int(i/16), i%16, :]
    position[i, -1] = i

prediction = model_rnn.predict(test_sequence)

print (prediction)

index = np.argmax(prediction)

print ("index=", index, "probability=", prediction[index], "sequence=", position[index])

#Representacao grafica das probabilidades

#para testar apenas o grafico criar prediction random comentar tudo para cima e descomentar a seguinte linha:
#prediction=np.random.rand(256,1)

#auxfunctions.probGraph(prediction)
