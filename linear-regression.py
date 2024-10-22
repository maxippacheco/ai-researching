import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
	def __init__(self, learning_rate=0.01, n_iterations=1000, method="SGD"):
		self.learning_rate = learning_rate
		self.n_iterations = n_iterations
		self.method = method
		self.theta = None

	def fit(self, X, y):
		m = len(y) # num of training samples
		X_b = np.c_[np.ones((m,1)), X] # bias term (intercept)


		if self.method == "SGD":
			# Initializing parameters (random)
			self.theta = np.random.randn(2,1)

			# For loss function we gonna use LMS algorithm (Stochaistic Gradient descent)
			# This method is ideally used for large dataset -> you can try B.G.D too.
			for iteration in range(self.n_iterations): # n_iterations = epochs
				for i in range(m):
					# This randomness helps prevent getting stuck in local minima
					random_index = np.random.randint(m)
					# Extracting a random training example
					x_i = X_b[random_index:random_index+1]
					y_i = y[random_index:random_index+1]
					
					# computes the predicted value for the randomly selected training example.
					prediction = x_i.dot(self.theta) # dot product as they are vectors
					error = y_i - prediction # diff between predicted and real value
					# The transpose allows us to calculate the gradients for all parameters in one step.
					# gradient= −2*(y−y^)*x
					gradients = -2 * x_i.T.dot(error) # taking the parcial derivative and multiplying by -2 to minimize the cost
					# Updating the parameters by moving them in the direction opposite to the gradient
					# self.theta -= ensures that we subtract the gradient to minimize the cost function
					self.theta -= self.learning_rate * gradients
		elif self.method == "BGD":
			# Initialize parameters randomly
			self.theta = np.random.randn(X_b.shape[1], 1)

			# Batch Gradient Descent
			for iteration in range(self.n_iterations):
				predictions = X_b.dot(self.theta)
				errors = y - predictions
				gradients = -2 / m * X_b.T.dot(errors)
				self.theta -= self.learning_rate * gradients
		elif self.method == "LSM":
			# Least Squares Method using the Normal equation
			self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

	def predict(self, X):
		X_b = np.c_[np.ones((X.shape[0], 1)), X] # Adding bias term
		return X_b.dot(self.theta) # Return predictions

np.random.seed(0)

# synthetic data
X = 2 * np.random.rand(100, 1)  # 100 random points between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# plotting
# plt.scatter(X, y)
# plt.xlabel("X")
# plt.ylabel("y")
# plt.title("Synthetic Data for Linear Regression (LMS)")
# plt.show()

# Choose the learning method
method = "SGD"  # Options: "SGD", "BGD", "LSM"

# Training the custom model using the selected method
model = LinearRegression(learning_rate=0.01, n_iterations=1000, method=method)
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Plot the synthetic data and the linear regression prediction
plt.scatter(X, y, label="Data")
plt.plot(X, predictions, color='red', label="Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Linear regression fit using {method}")
plt.legend()
plt.show()
