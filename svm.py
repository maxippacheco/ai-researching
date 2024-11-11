import numpy as np
import matplotlib.pyplot as plt

# to compute similarity between data points
def linear_kernel(x1, x2):
	return np.dot(x1, x2)

# SMO - Sequential Minimal optimization
def svm_optimization(X, y, kernel, C=1.0, tol=1e-3, max_passes=5):
	# samples - features
	m, n = X.shape
	# Lagrange multipliers for each training example, initially set to zero
	alphas = np.zeros(m)
	# bias term
	b = 0
	passes = 0
	
	while passes < max_passes:
		num_changed_alphas = 0
		for i in range(m):
			# Calculate the error term for the i-th training point
			Ei = np.sum(alphas * y * kernel(X[i], X.T)) + b - y[i]
			# This checks the KKT conditions, which determine whether the current point violates optimality
			if (y[i]*Ei < -tol and alphas[i] < C) or (y[i]*Ei > tol and alphas[i] > 0):
				# Randomly selects a second point, j, to pair with i for updating the Lagrange multipliers.
				j = np.random.choice([k for k in range(m) if k != i])
				Ej = np.sum(alphas * y * kernel(X[j], X.T)) + b - y[j]
				
				alpha_i_old, alpha_j_old = alphas[i], alphas[j]
				# Compute the bounds L and H for alphas[j]
				L, H = (max(0, alphas[j] - alphas[i]), min(C, C + alphas[j] - alphas[i])) if y[i] != y[j] else (max(0, alphas[i] + alphas[j] - C), min(C, alphas[i] + alphas[j]))
				
				if L == H:
					continue
				
				"""
				A value used in the SMO algorithm that measures the curvature of the objective function with respect to alphas[i] and alphas[j]. 
				If eta is non-negative, the update step is skipped because it would not reduce the objective function.
				"""
				eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])
				
				if eta >= 0:
					continue
				
				# updating alphas
				alphas[j] -= y[j] * (Ei - Ej) / eta
				alphas[j] = np.clip(alphas[j], L, H)
				
				if abs(alphas[j] - alpha_j_old) < 1e-5:
					continue
				
				alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])
				
				# updating bias terms
				b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * kernel(X[i], X[i]) - y[j] * (alphas[j] - alpha_j_old) * kernel(X[i], X[j])
				b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * kernel(X[i], X[j]) - y[j] * (alphas[j] - alpha_j_old) * kernel(X[j], X[j])
				
				if 0 < alphas[i] < C:
					b = b1
				elif 0 < alphas[j] < C:
					b = b2
				else:
					b = (b1 + b2) / 2
				
				# If any alphas were changed in the loop, num_changed_alphas is incremented
				num_changed_alphas += 1
		
		if num_changed_alphas == 0:
			# If no alphas are updated, the passes counter increments.
			passes += 1
		else:
			passes = 0

	return alphas, b

class SVM:
	def __init__(self, kernel=linear_kernel, C=1.0):
		self.kernel = kernel
		self.C = C
		self.alphas = None
		self.b = None
		self.support_vectors = None
		self.support_vectors_labels = None

	def fit(self, X, y):
		self.X = X
		self.y = y
		self.alphas, self.b = svm_optimization(X, y, self.kernel, self.C)

		support_vector_indices = np.where(self.alphas > 1e-5)[0]
		self.support_vectors = X[support_vector_indices]
		self.support_vectors_labels = y[support_vector_indices]
		self.alphas = self.alphas[support_vector_indices]

	def predict(self, X):
		y_pred = np.sum((self.alphas * self.support_vectors_labels)[:, np.newaxis] * self.kernel(self.support_vectors, X.T), axis=0) + self.b
		return np.sign(y_pred)
	
# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)


# Usage example
svm = SVM()
svm.fit(X, y)
y_pred = svm.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.2f}")

def plot_decision_boundary(svm, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                s=80, facecolors='none', edgecolors='k')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(svm, X, y)
