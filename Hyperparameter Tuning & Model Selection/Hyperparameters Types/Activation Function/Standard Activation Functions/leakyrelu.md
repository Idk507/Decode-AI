# Leaky ReLU Activation Function

## Mathematical Definition

Leaky ReLU introduces a small slope for negative inputs to address the dying ReLU problem:

```
LeakyReLU(x) = { x      if x ≥ 0
                { αx     if x < 0
```

Where `α` is a small positive constant (typically `α ≈ 0.01`).

## Mathematical Properties

- **Range**: `(-∞, ∞)`
- **Derivative**:
  ```
  d/dx LeakyReLU(x) = { 1    if x ≥ 0
                       { α    if x < 0
  ```
- **Behavior**: Linear for positive inputs, scaled linear for negative inputs
- **Non-Differentiable at x = 0**: Handled similarly to ReLU in practice

## Use Cases

- **Hidden Layers**: Used in place of ReLU to prevent dying neurons, especially in deep networks
- **Example**: Deep convolutional networks where ReLU causes too many dead neurons

## Advantages

- **Prevents Dying ReLU**: The small slope (`αx`) for negative inputs ensures gradients are non-zero, allowing neurons to recover
- Retains ReLU's computational efficiency and sparsity (to a lesser extent)
- Simple to implement

## Problems

- **Fixed α**: The choice of `α` (e.g., 0.01) is heuristic and may not be optimal for all tasks
- **Not Zero-Centered**: Like ReLU, outputs are not symmetric around 0
- **Unbounded Output**: Similar to ReLU, large outputs can cause instability

## Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Leaky ReLU function
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

# Derivative of Leaky ReLU
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x >= 0, 1, alpha)

# Generate input values
x = np.linspace(-10, 10, 100)
y = leaky_relu(x)
y_deriv = leaky_relu_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Leaky ReLU Function', color='blue')
plt.plot(x, y_deriv, label='Leaky ReLU Derivative', color='red', linestyle='--')
plt.title('Leaky ReLU Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

### Code Explanation

- **Leaky ReLU Function**: Outputs `x` for `x ≥ 0` and `αx` for `x < 0`, with `α = 0.01`
- **Derivative**: Outputs 1 for `x ≥ 0` and `α` for `x < 0`
- **Visualization**: The plot shows a small negative slope for `x < 0`, addressing the dying ReLU problem, and the derivative is a step function with a small value (`α`) for negative inputs

## Dependencies

```bash
pip install numpy matplotlib
```

## Usage Example

```python
# Example usage
import numpy as np

# Input data
x = np.array([-2, -1, 0, 1, 2])

# Apply Leaky ReLU
output = leaky_relu(x, alpha=0.01)
print(f"Input: {x}")
print(f"Output: {output}")
# Output: [-0.02 -0.01  0.    1.    2.  ]
```

## Key Features

- ✅ Solves the dying ReLU problem
- ✅ Computationally efficient  
- ✅ Simple implementation
- ⚠️ Fixed slope parameter
- ⚠️ Not zero-centered
- ⚠️ Unbounded output range
