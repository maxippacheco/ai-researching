from svm import SVM, plot_decision_boundary
import numpy as np

# Radial Basis Function (RBF) kernel, which is commonly used for non-linear classification problems
# Kernel Function is used to transform n-dimensional input to m-dimensional input, where m is much higher than n then find the dot product in higher dimensional efficiently
def rbf_kernel(x1, x2, gamma=1.0):
	return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis=2)**2)

# Generate non-linearly separable data
np.random.seed(0)
X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int) * 2 - 1

# Train SVM with RBF kernel
svm_rbf = SVM(kernel=lambda x1, x2: rbf_kernel(x1, x2, gamma=0.5), C=1.0)
svm_rbf.fit(X, y)

# Plot decision boundary
plot_decision_boundary(svm_rbf, X, y)
