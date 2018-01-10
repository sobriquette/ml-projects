"""
mnist.py
---------

Implement a class Network to calculate the stochastic gradient descent
for a feedforward neural network using backpropagation.

Source: http://neuralnetworksanddeeplearning.com/chap1.html
"""

import network

# Third-party libraries
from keras.datasets import mnist
import numpy as np

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
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# Reformat data so it can be consumed efficiently by neural network
	training_inputs = [np.reshape(x, (784, 1)) for x in x_train]
	training_results = [vectorized_result(y) for y in y_train]
	training_data = list(zip(training_inputs, training_results))

	test_inputs = [np.reshape(x, (784, 1)) for x in x_test]
	test_data = list(zip(test_inputs, y_test))

	net = network.Network([784, 30, 10])

	net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



