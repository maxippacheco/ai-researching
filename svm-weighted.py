import numpy as np
from svm import SVM, linear_kernel, svm_optimization, plot_decision_boundary

class WeightedSVM(SVM):
	def __init__(self, kernel=linear_kernel, C=1.0, class_weight=None):
		super().__init__(kernel, C)
		self.class_weight= class_weight

	def fit(self, X, y):
		if self.class_weight is None:
			# If no class weights, assigns equal weight to all samples
			self.class_weight = np.ones(len(y))
		else:
      # If class weights are provided, sets sample weights according to class labels in y
			sample_weights = np.array([self.class_weight[label] for label in y])

		self.X = X
		self.y = y

    # Calls the SVM optimization function with adjusted C values based on sample weights
		self.alphas, self.b = svm_optimization(X, y, self.kernel, self.C * sample_weights)

		# Finds indices of support vectors where alpha values are above a small threshold
		support_vector_indices = np.where(self.alphas > 1e-5)[0]
		# Stores the support vectors and their corresponding labels
		self.support_vectors = X[support_vector_indices]
		self.support_vector_labels = y[support_vector_indices]
		# Keeps only the non-zero alphas corresponding to support vectors
		self.alphas = self.alphas[support_vector_indices]

# Example usage with imbalanced data
np.random.seed(0)
X_imbalanced = np.random.randn(300, 2)
y_imbalanced = np.concatenate([np.ones(250), -np.ones(50)])

svm_weighted = WeightedSVM(class_weight={1: 1, -1: 5})
svm_weighted.fit(X_imbalanced, y_imbalanced)
plot_decision_boundary(svm_weighted, X_imbalanced, y_imbalanced)
