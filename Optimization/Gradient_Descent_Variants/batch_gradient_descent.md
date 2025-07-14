# Batch Gradient Descent (BGD) - Complete Guide

## Table of Contents
- [Mathematical Concepts](#mathematical-concepts)
- [Python Implementation](#python-implementation)
- [Usage](#usage)
- [Requirements](#requirements)

## Mathematical Concepts

### Key Idea
- **Objective**: Minimize the loss function J(θ) where θ represents model parameters (weights).
- **Approach**: Compute the gradient of the loss w.r.t. all training examples and update weights in the opposite direction.

### Algorithm Steps

1. **Compute Loss** (for entire training set):
   ```
   J(θ) = (1/m) Σ[i=1 to m] Loss(h_θ(x^(i)), y^(i))
   ```
   - `m` = number of training samples
   - `h_θ(x)` = predicted output
   - `y` = true label

2. **Compute Gradient** (average gradient over all training samples):
   ```
   ∇_θ J(θ) = (1/m) Σ[i=1 to m] ∇_θ Loss(h_θ(x^(i)), y^(i))
   ```

3. **Update Weights** (using learning rate α):
   ```
   θ = θ - α · ∇_θ J(θ)
   ```

### Example: Linear Regression

- **Hypothesis**: `h_θ(x) = θ_0 + θ_1 x`

- **Loss (MSE)**:
  ```
  J(θ_0, θ_1) = (1/2m) Σ[i=1 to m] (h_θ(x^(i)) - y^(i))²
  ```

- **Gradients**:
  ```
  ∂J/∂θ_0 = (1/m) Σ[i=1 to m] (h_θ(x^(i)) - y^(i))
  
  ∂J/∂θ_1 = (1/m) Σ[i=1 to m] (h_θ(x^(i)) - y^(i)) x^(i)
  ```

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
```

### Batch Gradient Descent Implementation

```python
def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Perform Batch Gradient Descent for Linear Regression
    
    Parameters:
    X: Input features (m x n matrix)
    y: Target values (m x 1 vector)
    learning_rate: Learning rate (alpha)
    n_iterations: Number of iterations
    
    Returns:
    theta: Optimized parameters
    cost_history: Cost function values during training
    """
    m = len(y)  # Number of training examples
    
    # Add bias term (intercept)
    X_b = np.c_[np.ones((m, 1)), X]  # Add x0 = 1 to each instance
    
    # Initialize parameters randomly
    theta = np.random.randn(2, 1)
    
    cost_history = []
    
    for iteration in range(n_iterations):
        # Forward pass: compute predictions
        predictions = X_b.dot(theta)
        
        # Compute cost (MSE)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)
        
        # Compute gradients
        gradients = (1/m) * X_b.T.dot(predictions - y)
        
        # Update parameters
        theta = theta - learning_rate * gradients
    
    return theta, cost_history

# Train the model
theta_optimal, cost_history = batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)

print(f"Optimal parameters:")
print(f"θ_0 (intercept): {theta_optimal[0][0]:.4f}")
print(f"θ_1 (slope): {theta_optimal[1][0]:.4f}")
```

### Visualization

```python
# Plot the results
plt.figure(figsize=(12, 4))

# Plot 1: Training data with fitted line
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.6)
X_plot = np.linspace(0, 2, 100).reshape(-1, 1)
X_plot_b = np.c_[np.ones((100, 1)), X_plot]
y_predict = X_plot_b.dot(theta_optimal)
plt.plot(X_plot, y_predict, 'r-', linewidth=2, label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with BGD')
plt.legend()
plt.grid(True)

# Plot 2: Cost function over iterations
plt.subplot(1, 2, 2)
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function During Training')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Usage

1. **Clone or download** this repository
2. **Install requirements**: `pip install numpy matplotlib`
3. **Run the script** to see BGD in action on synthetic data
4. **Modify parameters** (learning rate, iterations) to see their effects

## Requirements

```
numpy>=1.19.0
matplotlib>=3.3.0
```

## Key Advantages of BGD

- **Stable convergence**: Uses entire dataset for each update
- **Guaranteed convergence**: For convex loss functions
- **Optimal solution**: Finds global minimum for convex problems

## Key Disadvantages of BGD

- **Computational cost**: Expensive for large datasets
- **Memory requirements**: Needs entire dataset in memory
- **Slow updates**: One update per full dataset pass

## Variations

- **Stochastic Gradient Descent (SGD)**: Uses one sample per update
- **Mini-batch Gradient Descent**: Uses small batches of samples
- **Momentum-based methods**: Add momentum to gradient updates

## Contributing

Feel free to contribute improvements, additional examples, or extensions to other algorithms!

## License

This project is open source and available under the MIT License.
