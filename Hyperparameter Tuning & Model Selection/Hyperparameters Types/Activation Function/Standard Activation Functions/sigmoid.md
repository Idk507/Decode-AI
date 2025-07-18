# Sigmoid (Logistic) Activation Function

## Mathematical Definition

The sigmoid function, also known as the logistic function, maps any real number to the range (0, 1). Its formula is:

```
σ(x) = 1 / (1 + e^(-x))
```

Where:
- `x` is the input (a real number)
- `e` is the base of the natural logarithm (≈ 2.71828)
- The output `σ(x)` lies between 0 and 1

## Mathematical Properties

- **Range**: (0, 1), making it ideal for probabilities
- **Derivative**:
  ```
  d/dx σ(x) = σ(x) · (1 - σ(x))
  ```
  This derivative is simple to compute, as it uses the sigmoid output itself.

- **Behavior**:
  - As `x → ∞`, `σ(x) → 1`
  - As `x → -∞`, `σ(x) → 0`
  - At `x = 0`, `σ(0) = 0.5`

- **S-shape**: The function is smooth and S-shaped (sigmoidal), providing non-linearity
- **Not Zero-Centered**: The output is always positive, which can cause issues in gradient-based optimization

## Use Cases

- **Binary Classification**: Used in the output layer of neural networks for binary classification tasks, where the output represents a probability (e.g., probability of class 1)
- **Logistic Regression**: The sigmoid function is the core of logistic regression models
- **Example**: Predicting whether an email is spam (1) or not spam (0)

## Advantages

- Outputs interpretable probabilities (0 to 1)
- Smooth and differentiable, enabling gradient-based optimization
- Simple derivative, which is computationally efficient

## Problems

- **Vanishing Gradients**: For large positive or negative inputs (`|x|` is large), the derivative `σ'(x)` approaches 0, slowing down learning in deep networks (gradients vanish)
- **Not Zero-Centered**: Outputs are always positive, which can lead to inefficient gradient updates during backpropagation (e.g., zig-zagging in gradient descent)
- **Computationally Expensive**: The exponential function `e^(-x)` is costly to compute compared to simpler operations

## Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Generate input values
x = np.linspace(-10, 10, 100)
y = sigmoid(x)
y_deriv = sigmoid_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sigmoid Function', color='blue')
plt.plot(x, y_deriv, label='Sigmoid Derivative', color='red', linestyle='--')
plt.title('Sigmoid Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

### Code Explanation

- **Sigmoid Function**: Computes `σ(x) = 1 / (1 + e^(-x))` using NumPy's `exp` function
- **Derivative**: Computes `σ'(x) = σ(x) · (1 - σ(x))`, which is used in backpropagation
- **Visualization**: Plots the sigmoid function (S-shaped) and its derivative (bell-shaped, peaking at 0.25 when `x = 0`)
- **Range Check**: The plot confirms the output lies between 0 and 1, and the derivative is small for large `|x|`, illustrating the vanishing gradient problem

## Dependencies

```bash
pip install numpy matplotlib
```

## Usage Example

```python
# Example usage
x_test = np.array([-2, -1, 0, 1, 2])
output = sigmoid(x_test)
print(f"Input: {x_test}")
print(f"Sigmoid Output: {output}")
print(f"Derivative: {sigmoid_derivative(x_test)}")
```

Expected output:
```
Input: [-2 -1  0  1  2]
Sigmoid Output: [0.11920292 0.26894142 0.5        0.73105858 0.88079708]
Derivative: [0.10499359 0.19661193 0.25       0.19661193 0.10499359]
```
