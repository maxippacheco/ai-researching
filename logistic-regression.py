import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

class LogisticRegression:
	def sigmoid(self, z):
		return 1/(1 + np.e**(-z)) # hypotheses function
	
	# it penalizes wrong predictions and upload the parameters to maximize the probability of classifying correctly the instances
	def cost_function(self, X, y, weights):
		z = np.dot(X, weights)
		# positive class(1)
		predict_1 = y * np.log(self.sigmoid(z))
		# negative class(0)
		predict_0 = (1 - y) * np.log(1 - self.sigmoid(z))

		# we use -sum(...) cause the cross entropy is the negative value for the log-probabilities sum.
		# we divide for len(X) to get the avg cost per train exs.
		# Cross-entropy loss, also known as logarithmic loss or log loss, is a popular loss function used in machine learning 
		# to measure the performance of a classification model. It quantifies the difference between the predicted probability
		# distribution and the true distribution of the data. -> basically the likelihood maximized function lol :D(log one).
		return -sum(predict_1 + predict_0) / len(X)
	
	def fit(self, X, y, epochs=25, learning_rate=0.05):
		# value of the cost function at each epoch
		loss = []
		# Initializes the weights randomly with the same number of elements as the number of features
		weights = rand(X.shape[1])
		N = len(X)

		for _ in range(epochs):
			# Gradient descent
			# Calculates the predicted probabilities
			y_hat = self.sigmoid(np.dot(X, weights))
			# difference between predicted and actual values
			error = y_hat - y
			# Updates the weights
			# np.dot(...) computes the gradient of the cost function with respect to weights
			# learning_rate * ... / N scales the gradient by learning_rate and normalizes it by N.
			weights -= learning_rate * np.dot(X.T, error) / N
			# Saving progress
			loss.append(self.cost_function(X, y, weights))
		
		self.weights = weights
		self.loss = loss

	
	def predict(self, X):
		# Prediction using sigmoid function
		z = np.dot(X, self.weights)

		# Returning binary result

		return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]
	

# Example data generation
np.random.seed(0)
n_samples = 100

# Generate random features
X = np.c_[np.ones((n_samples, 1)), 2 * rand(n_samples, 1)]  # Bias term + one feature

# Generate binary labels based on a threshold for simplicity
true_weights = np.array([0.5, 2])  # True weights for a synthetic decision boundary
y = (np.dot(X, true_weights) + 0.1 * rand(n_samples) > 0).astype(int)  # Add some noise

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y, epochs=100, learning_rate=0.1)

# Plot the loss over epochs
plt.plot(range(len(model.loss)), model.loss)
plt.xlabel('Epochs')
plt.ylabel('Cost (Cross-Entropy Loss)')
plt.title('Cost Function over Epochs')
plt.show()

# Test the model's predictions
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100:.2f}%")
