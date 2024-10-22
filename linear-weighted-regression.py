import numpy as np
import matplotlib.pyplot as plt

class LocallyWeightedLinearRegression:
	def __init__(self, tau=0.1):
		"""
    tau: Bandwidth parameter that controls how fast the weights decay with distance.
    """
		self.tau = tau
		self.theta = None

	def _compute_weights(self, X_b, x_query):
		"""
    Compute the weight matrix W based on the distance between each point in X_b and x_query.
    W is a diagonal matrix where W[i, i] is the weight of the ith training example.
    """
		m = X_b.shape[0]
		W = np.eye(m)
		for i in range(m):
			diff = X_b[i] - x_query # Distance between the query point and each training example
			W[i, i] = np.exp(-np.dot(diff, diff.T) / (2*self.tau**2)) # Gaussian kernel
		# return the weight
		return W
	
	def predict(self, X, y, x_query):
		"""
    Given a single query point x_query, predict the output using locally weighted linear regression.
    """
		m = X.shape[0]
		X_b = np.c_[np.ones((m, 1)), X] # Adding bias term

		# Adding bias term to the query point
		x_query_b = np.r_[1, x_query]

		# Compute the weights
		W = self._compute_weights(X_b, x_query_b)

		# Normal Equation with weights
		theta = np.linalg.inv(
			X_b.T.dot(W).dot(X_b)
		).dot(X_b.T).dot(W).dot(y)
		
		# Predict the output for the query point
		return x_query_b.dot(theta)

# Example of LWLR usage:
np.random.seed(0)


# Synthetic data
X = 2 * np.random.rand(100, 1)  # 100 random points between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Create an instance of LocallyWeightedLinearRegression
lwlr = LocallyWeightedLinearRegression(tau=0.1)

# Predict for a specific query point
x_query = np.array([1.2])  # A single query point where we want to make a prediction
y_pred = lwlr.predict(X, y, x_query)

print(f"Predicted value at x_query={x_query[0]}: {y_pred[0]}")

# Plotting the synthetic data
plt.scatter(X, y, label="Data")
plt.plot(x_query, y_pred, 'ro', label=f"Prediction at x_query={x_query[0]}")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Locally Weighted Linear Regression (LWLR)")

# Show the prediction at x_query in red
plt.legend()
plt.show()