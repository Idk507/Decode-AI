# Hyperbolic Tangent (Tanh) Activation Function

## Mathematical Definition

The hyperbolic tangent function maps inputs to the range `(-1, 1)`:

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

Alternatively, it can be expressed in terms of the sigmoid function:

```
tanh(x) = 2σ(2x) - 1
```

## Mathematical Properties

- **Range**: `(-1, 1)`, making it zero-centered
- **Derivative**:
  ```
  d/dx tanh(x) = 1 - tanh²(x)
  ```
  The derivative is simple and depends on the output of the tanh function
- **Behavior**:
  - As `x → ∞`, `tanh(x) → 1`
  - As `x → -∞`, `tanh(x) → -1`
  - At `x = 0`, `tanh(0) = 0`
- **S-shape**: Like sigmoid, it's smooth and sigmoidal but centered at 0
- **Zero-Centered**: Unlike sigmoid, the output is symmetric around 0, which helps with gradient updates

## Use Cases

- **Hidden Layers**: Preferred over sigmoid in hidden layers because it's zero-centered, leading to faster convergence in gradient-based optimization
- **Example**: Used in recurrent neural networks (RNNs) or earlier feedforward networks

## Advantages

- **Zero-Centered**: Outputs are symmetric around 0, which helps gradients flow better during backpropagation compared to sigmoid
- **Smooth and differentiable**
- **Maps inputs to a wider range** `(-1, 1)` than sigmoid, providing stronger gradients

## Problems

- **Vanishing Gradients**: Like sigmoid, the derivative approaches 0 for large `|x|`, causing slow learning in deep networks
- **Computationally Expensive**: Involves exponentials, similar to sigmoid

## Implementation in Python

```python
import numpy as np
import matplotlib.pyplot as plt

# Tanh function
def tanh(x):
    return np.tanh(x)

# Derivative of tanh
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Generate input values
x = np.linspace(-10, 10, 100)
y = tanh(x)
y_deriv = tanh_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Tanh Function', color='blue')
plt.plot(x, y_deriv, label='Tanh Derivative', color='red', linestyle='--')
plt.title('Tanh Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

## Code Explanation

- **Tanh Function**: Uses NumPy's built-in `tanh` for efficiency
- **Derivative**: Computes `1 - tanh²(x)`, which is used in backpropagation
- **Visualization**: The plot shows the S-shaped tanh function (ranging from -1 to 1) and its derivative (peaking at 1 when `x = 0`)
- **Zero-Centered**: The plot confirms the output is symmetric around 0, and the derivative highlights the vanishing gradient issue for large `|x|`

## Requirements

```
numpy
matplotlib
```

## Installation

```bash
pip install numpy matplotlib
```

## Usage

Run the Python script to visualize the tanh activation function and its derivative. The plot will show:
- The tanh function in blue (S-shaped curve from -1 to 1)
- The derivative in red dashed line (bell-shaped curve peaking at 1)
