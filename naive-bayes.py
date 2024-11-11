from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Multi-class naive bayes classifier
class NaiveBayesClassifier:
	def fit(self, X, y):
		# separate data by class
		self.classes = np.unique(y)
		self.means = {}
		self.variances = {}
		self.priors = {} # Prior probabilities for each class

		for c in self.classes:
			X_c = X[y == c] # Get all instances of class c
			self.means[c] = X_c.mean(axis=0)
			self.variances[c] = X_c.var(axis=0)
			self.priors[c] = X_c.shape[0] / X.shape[0]
	
	def _calculate_likelihood(self, x, mean, var):
		"""
		Measures how well a statistical model explains observed data by calculating the probability of seeing 
		that data under different parameter values of the model
		"""
		# Gaussian prob density fn
		eps = 1e-6
		coef = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
		exp = np.exp(-(np.power(x - mean, 2) / (2 * var + eps)))

		return coef * exp
	
	def _calculate_posterior(self, x):
		posteriors = []

		# Calculate posterior probability for each class
		for c in self.classes:
			prior = np.log(self.priors[c]) # log prior
			class_conditional = np.sum(np.log(self._calculate_likelihood(x, self.means[c], self.variances[c])))
			posterior = prior + class_conditional
			posteriors.append(posterior)
		
		return self.classes[np.argmax(posteriors)]
	
	def predict(self, X):
		return [self._calculate_posterior(x) for x in X]

# note:
"""
Laplace smoothing is generally used when you're dealing with discrete data (categorical features), where it's possible for certain combinations of features and 
classes to never appear in the training set. In such cases, applying smoothing ensures that you don't get zero probabilities for these unseen combinations 
during inference.
However, for continuous data, like the features in the Iris dataset (sepal length, sepal width, etc.), 
Laplace smoothing is not necessary for the likelihoods because the Gaussian distribution (which is used here) inherently provides non-zero probabilities
for all possible feature values, even for unseen ones. In the case of Gaussian Naive Bayes, applying Laplace smoothing to the prior probabilities is 
typically what people refer to when smoothing is applied in a continuous setting.
"""


# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize the classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Predict on test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
