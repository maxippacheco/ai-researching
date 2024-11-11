# Machine Learning Algorithms Implementations

This branch contains implementations of various machine learning techniques, including:

- **Linear Regression Techniques:**
  - Stochastic Gradient Descent (SGD)
  - Batch Gradient Descent (BGD)
  - Least Squares Method (LSM)
  - Locally Weighted Regression (LWR)
  - Logistic Regression
  - Softmax Regression

- **Support Vector Machine (SVM) Techniques:**
  - SVM with Sequential Minimal Optimization (SMO)
  - SVM with Non-Linear Kernel
  - Weighted SVM

- **Naive Bayes Classifier**

## Overview

This project provides various approaches for performing regression and classification, covering both linear and non-linear models, each with unique advantages and applications.

## Techniques Implemented

### Linear Regression Techniques

1. **Stochastic Gradient Descent (SGD):** An iterative method for optimizing an objective function by updating model parameters with one training example at a time. This allows for faster convergence on large datasets.

2. **Batch Gradient Descent (BGD):** Computes the gradient of the cost function for the entire dataset at once, providing more stable updates but potentially slower convergence on large datasets.

3. **Least Squares Method (LSM):** Provides a direct solution to linear regression by minimizing the sum of the squared residuals. It is computationally efficient for small to medium-sized datasets.

4. **Locally Weighted Regression (LWR):** A non-parametric method that builds a linear model for each data point, weighting nearby points more heavily, allowing for flexible modeling of relationships that vary across the input space.

5. **Logistic Regression:** A statistical method used for binary classification, modeling the probability of a class or event. It applies the logistic function, yielding a probability between 0 and 1.

6. **Softmax Regression:** Extends logistic regression to multiple classes, using the softmax function to convert raw scores into probabilities for each class, making it useful for multi-class classification.

### Support Vector Machine (SVM) Techniques

1. **SVM with Sequential Minimal Optimization (SMO):** An efficient algorithm to solve the quadratic optimization problem in SVMs, allowing for quick convergence to the optimal solution.

2. **SVM with Non-Linear Kernel:** Extends SVM to handle non-linear relationships by using kernel functions (e.g., RBF, polynomial) that transform data into a higher-dimensional space.

3. **Weighted SVM:** A variation of SVM that assigns different weights to classes, making it useful for imbalanced datasets by giving more importance to underrepresented classes.

### Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming independence between features. It is simple yet effective, especially for text classification tasks and situations with high-dimensional data.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/maxippacheco/ai-researching/
cd <repository-directory>
pip install numpy matplotlib
```