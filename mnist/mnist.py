"""
mnist.py
---------

Train a model to be able to identify handwritten digits

Source: https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-4
"""

import network

# Third-party libraries
import numpy as np
np.random.seed(123) # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

def vectorized_result(j):
	"""
	Return a 10-dimensional unit vector with a 1.0 in the jth position
	and zeroes elsewhere. Convert a digit 0-9 into corresponding
	desired output from the NN.
	"""

	e = np.zeros((10, 1))
	e[j] = 1.0
	return e


if __name__=="__main__":
	# Load pre-shuffled MNIST data into train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	 
	# Preprocess input data
	X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
	X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	 
	# Preprocess class labels
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_test = np_utils.to_categorical(y_test, 10)
	 
	# Define model architecture
	model = Sequential()
	 
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28), dim_ordering='th'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	 
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	 
	# Compile model
	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	 
	# Fit model on training data
	model.fit(X_train, Y_train, 
	          batch_size=32, epochs=10, verbose=1)
	 
	# Evaluate model on test data
	score = model.evaluate(X_test, Y_test, verbose=0)

	# Using custom NN
	# training_inputs = [np.reshape(x, (784, 1)) for x in x_train]
	# training_results = [vectorized_result(y) for y in y_train]
	# training_data = list(zip(training_inputs, training_results))

	# test_inputs = [np.reshape(x, (784, 1)) for x in x_test]
	# test_data = list(zip(test_inputs, y_test))

	# net = network.Network([784, 30, 10])
	# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



