# Interesting fact:
"""
Softmax regression network (SRN)
A neural network that uses the softmax activation function to treat regression as a classification problem. 
The output layer of the ANN has one neuron for each bin, and the softmax function ensures that the outputs sum to 1. 
"""

import numpy as np

# logits - the outputs of a neural network before the activation function is applied
# Softmax Function: Converts logits into probabilities.
def softmax(logits):
	exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
	return exp_logits / exp_logits.sum(axis=1, keepdims=True)

# Cross-Entropy Loss: Measures the performance of the model by comparing predicted probabilities to true labels.
# The cross-entropy loss function is smooth and differentiable, which is essential for optimizing the model using gradient descent.
def cross_entropy_loss(y_true, y_pred):
	y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
	return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Gradient Computation: Computes the gradient of the loss function with respect to the weights.
def compute_gradients(X, y_true, y_pred):
	return np.dot(X.T, (y_pred - y_true)) / y_true.shape[0]

class SoftmaxRegression:
	def __init__(self, learning_rate=0.01, epochs=1000):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.W = None

	def fit(self, X, y):
		num_samples, num_features = X.shape
		num_classes = y.shape[1]

		# Initializing weights as multidim array features x classes
		self.W = np.zeros((num_features, num_classes))

		# Training Loop: Updates the weights using gradient descent based on the computed gradients.
		for _ in range(self.epochs):
			# compute logits
			logits = np.dot(X, self.W)

			# comput probs - softmax is the activation fn
			y_pred = softmax(logits)

			# Compute loss - quantifies the difference between prediction and expected value
			loss = cross_entropy_loss(y, y_pred)

			# Compute gradients
			gradients = compute_gradients(X, y, y_pred)

			# Update weights
			self.W -= self.learning_rate * gradients

			# Printing loss
			if _ % 100 == 0:
				print(f'Iteration {_}, Loss: {loss}')

	def predict(self, X):
		logits = np.dot(X, self.W)
		y_pred = softmax(logits)
		return np.argmax(y_pred, axis=1)
	
# Generate synthetic data
np.random.seed(0)
num_samples = 200
num_features = 2
num_classes = 3

X = np.random.randn(num_samples, num_features)
y = np.zeros((num_samples, num_classes))
for i in range(num_samples):
    y[i, np.random.randint(0, num_classes)] = 1

# Train the model
model = SoftmaxRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

# Predict on training data
predictions = model.predict(X)
print(f'Predictions: {predictions}')
