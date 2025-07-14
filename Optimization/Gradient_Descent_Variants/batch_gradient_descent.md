# Batch Gradient Descent (BGD) - Complete Guide

## Mathematical Concepts

### Key Idea
- **Objective**: Minimize the loss function \( J(\theta) \) where \( \theta \) represents model parameters (weights).
- **Approach**: Compute the gradient of the loss w.r.t. all training examples and update weights in the opposite direction.

![Gradient Descent Visualization](https://github.com/user-attachments/assets/13c63369-0c36-4f92-9bb4-f8748ecc3d69)

### Algorithm Steps
1. **Compute Loss** (for entire training set):
   \[
   J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \text{Loss}(h_\theta(x^{(i)}), y^{(i)})
   \]
   - \( m \) = number of training samples
   - \( h_\theta(x) \) = predicted output
   - \( y \) = true label

2. **Compute Gradient** (average gradient over all training samples):
   \[
   \nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \text{Loss}(h_\theta(x^{(i)}), y^{(i)})
   \]

3. **Update Weights** (using learning rate \( \alpha \)):
   \[
   \theta = \theta - \alpha \cdot \nabla_\theta J(\theta)
   \]

### Example: Linear Regression
- **Hypothesis**: \( h_\theta(x) = \theta_0 + \theta_1 x \)
- **Loss (MSE)**:
  \[
  J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
  \]
- **Gradients**:
  \[
  \frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})
  \]
  \[
  \frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}
  \]

## Python Implementation

### Generating Synthetic Data
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (y = 2x + 5 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 random points between [0, 2]
y = 5 + 2 * X + np.random.randn(100, 1)  # Linear relation with noise

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Synthetic Linear Data")
plt.show()
