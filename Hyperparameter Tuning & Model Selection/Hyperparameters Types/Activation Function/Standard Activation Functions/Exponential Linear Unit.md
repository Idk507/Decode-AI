# Exponential Linear Unit (ELU) Activation Function

## Mathematical Definition

ELU uses an exponential function for negative inputs:

```
ELU(x) = {
    x                   if x ≥ 0
    α(e^x - 1)         if x < 0
}
```

Where `α` is a positive constant (typically `α = 1`).

## Mathematical Properties

- **Range**: `(-α, ∞)`
- **Derivative**:
  ```
  d/dx ELU(x) = {
      1           if x ≥ 0
      αe^x        if x < 0
  }
  ```
- **Behavior**: Linear for positive inputs, smooth exponential for negative inputs
- **Smooth Transition**: Unlike ReLU, ELU is differentiable everywhere, including at `x = 0`

## Use Cases

- **Hidden Layers**: Used in deep networks to avoid dying ReLU and improve convergence
- **Applications**: Computer vision or natural language processing tasks requiring robust performance

## Advantages

- **Avoids Dying ReLU**: Non-zero gradients for negative inputs
- **Smooth for Negative Inputs**: The exponential curve provides smoother transitions
- **Closer to Zero-Centered**: Negative outputs are bounded by `-α`, reducing bias in gradients

## Problems

- **Computationally Expensive**: The exponential function is slower than ReLU's max operation
- **Hyperparameter α**: The choice of `α` is heuristic
- **Unbounded Positive Outputs**: Like ReLU, large positive values can cause instability

## Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# ELU function
def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# Derivative of ELU
def elu_derivative(x, alpha=1.0):
    return np.where(x >= 0, 1, alpha * np.exp(x))

# Generate input values
x = np.linspace(-10, 10, 100)
y = elu(x)
y_deriv = elu_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='ELU Function', color='blue')
plt.plot(x, y_deriv, label='ELU Derivative', color='red', linestyle='--')
plt.title('ELU Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

### Code Explanation

- **ELU Function**: Outputs `x` for `x ≥ 0` and `α(e^x - 1)` for `x < 0`
- **Derivative**: Outputs `1` for `x ≥ 0` and `αe^x` for `x < 0`
- **Visualization**: The plot shows a smooth exponential curve for negative inputs, approaching `-α`, and a linear function for positive inputs

## Installation Requirements

```bash
pip install numpy matplotlib
```

## Usage Example

```python
import numpy as np

# Create sample data
x = np.array([-2, -1, 0, 1, 2])

# Apply ELU activation
activated = elu(x, alpha=1.0)
print(f"Input: {x}")
print(f"ELU Output: {activated}")

# Compute gradients
gradients = elu_derivative(x, alpha=1.0)
print(f"Gradients: {gradients}")
```

## References

- Used in deep learning architectures for improved gradient flow
- Particularly effective in convolutional neural networks and recurrent neural networks
- Alternative to ReLU family of activation functions
