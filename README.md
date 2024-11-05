# Linear Regression Implementations

This branch contains implementations of various linear regression techniques, including:

1. **Stochastic Gradient Descent (SGD)**
2. **Batch Gradient Descent (BGD)**
3. **Least Squares Method (LSM)**
4. **Locally Weighted Regression (LWR)**
5. **Logistic Regression**
6. **Softmax Regression**

## Overview

Linear regression is a fundamental statistical method used to model the relationship between a dependent variable and one or more independent variables. This project implements various approaches to perform regression, including linear and logistic regression techniques, each with its unique advantages and use cases.

### Techniques Implemented

#### 1. Stochastic Gradient Descent (SGD)
SGD is an iterative method for optimizing an objective function. It updates the model parameters using one training example at a time, which allows for faster convergence in large datasets.

#### 2. Batch Gradient Descent (BGD)
BGD computes the gradient of the cost function for the entire dataset and updates the model parameters in one go. While it is more stable than SGD, it can be slower for large datasets due to its reliance on the full dataset for each update.

#### 3. Least Squares Method (LSM)
LSM provides a direct solution to linear regression by minimizing the sum of the squares of the residuals (the differences between the observed and predicted values). This method is straightforward and computationally efficient for small to medium-sized datasets.

#### 4. Locally Weighted Regression (LWR)
LWR is a non-parametric method that builds a linear regression model for each point in the dataset, weighting nearby points more heavily than those further away. This approach allows for flexible modeling of relationships that vary over the input space.

#### 5. Logistic Regression
Logistic regression is a statistical method used for binary classification that models the probability of a certain class or event. It applies the logistic function to model the relationship between a dependent variable and one or more independent variables, resulting in a probability value between 0 and 1.

#### 6. Softmax Regression
Softmax regression is a generalization of logistic regression to multiple classes. It uses the softmax function to convert raw scores (logits) into probabilities for each class, ensuring that the probabilities sum to 1. This method is particularly useful for multi-class classification problems.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/maxippacheco/ai-researching/
cd <repository-directory>
pip install numpy matplotlib # or you can create your own virtual environment
```