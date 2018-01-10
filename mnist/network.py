"""
network.py
------------

Implement a class Network to calculate the stochastic gradient descent
for a feedforward neural network using backpropagation.

Source: http://neuralnetworksanddeeplearning.com/chap1.html
"""

import random
import numpy as np

class Network(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		""" Return the output of the network if "a" is input."""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)

		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		"""
		Train the neural network using mini-batch stochastic gradient descent. 
		The "training_data" is a list of tuples "(x, y)" representing the training inputs
		and the desired outputs.
		"""
		if test_data: 
			n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [ training_data[k : k + mini_batch_size] \
							 for k in range(0, n, mini_batch_size) ]
			if test_data:
				print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
			else:
				print("Epoch {0} complete".format(j))

	def update_mini_batch(self, mini_batch, eta):
		"""
		Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		"mini_batch" is a list of tuples (x, y) and "eta" is the learning rate.
		"""

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		self.weights = [w - (eta/len(mini_batch)) * nw \
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (eta/len(mini_batch)) * nb \
						for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# feedforward
		activation = x
		activations [x] # store activations layer by layer
		zs = [] #store all z vectors, layer by layer

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
				sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-1] = np.dot(delta, activations[-l - 1].transpose())

		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) \
						for (x, y) in test_data]

		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		return (output_activations - y)



def sigmoid(z):
	# Prevent overflow
	signal = np.clip(z, -500, 500)
	return 1.0 / (1.0 + np.exp(-signal))

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))
