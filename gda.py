"""
When we have a classification problem in which the input features x are continuous-valued random variables, 
we can then use the Gaussian Discriminant Analysis (GDA) model, which models p(x|y) using a multivariate normal distribution.

GDA is a family of generative algorithm that try to model p(x|y) (and p(y)). 
After modeling p(y) (called the class priors) and p(x|y), our algorithm can then use Bayes rule to
derive the posterior distribution on y given x.
"""

import numpy as np

class GDABinaryClassifier:
	def fit(self, X, y):
		# This line calculates the prior probability
		self.fi = y.mean()
		# is an array containing the mean vectors of each class
		self.u = np.array([ X[y==k].mean(axis=0) for k in [0,1]])

		X_u = X.copy()
		# Calculate covariances
		for k in [0,1]: X_u[y==k] -= self.u[k]
		self.E = X_u.T.dot(X_u) / len(y)
		# The pseudo-inverse is often used instead of the standard inverse to ensure stability in cases where the matrix may be singular
		self.invE = np.linalg.pinv(self.E)

		return self
	
	def predict(self, X):
		"""
		For each class i (0 or 1), it computes the probability of each sample in X belonging to class i by calling compute_prob. 
		The class with the higher probability is chosen as the predicted class (using np.argmax), giving the final prediction array.
		"""
		return np.argmax([self.compute_prob(X, i) for i in range(len(self.u))], axis=0)
	
	def compute_prob(self, X, i):
		"""
			This line extracts the mean vector u for class i and calculates phi, the class prior probability for class i based on self.fi.
		"""
		u, phi = self.u[i], ((self.fi)**i * (1 - self.fi)**(1 - i))
		"""
			This line calculates the probability density function (PDF) of each sample in X for class i
		"""
		return np.exp(-1.0 * np.sum((X-u).dot(self.invE)*(X-u), axis=1)) * phi
	
	def score(self, X, y):
		return (self.predict(X) == y).mean()

# Testing the model	
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
model = GDABinaryClassifier().fit(X, y)
prediction = model.predict(X)
score = model.score(X, y)
score = score * 100
print(f"Score: {score:.2f}")



