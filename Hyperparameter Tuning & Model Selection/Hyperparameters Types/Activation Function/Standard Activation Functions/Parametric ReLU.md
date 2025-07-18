# Parametric ReLU (PReLU) Activation Function

## Mathematical Definition

PReLU is similar to Leaky ReLU, but α is a learnable parameter:

```
PReLU(x) = {
    x      if x ≥ 0
    αx     if x < 0
}
```

Where α is learned during training, typically initialized to a small value like 0.25.

## Mathematical Properties

- **Range**: (-∞, ∞)
- **Derivative**:
  ```
  d/dx PReLU(x) = {
      1    if x ≥ 0
      α    if x < 0
  }
  ```
  Additionally, the gradient with respect to α is computed during backpropagation.
- **Behavior**: Like Leaky ReLU, but α adapts to the data.

## Use Cases

- **Deep Networks**: Used in architectures like ResNet, where learning the negative slope improves performance
- **Example**: Image recognition tasks in very deep CNNs

## Advantages

- **Learnable α**: Adapts to the data, potentially improving performance over fixed α in Leaky ReLU
- Prevents dying ReLU problem
- Retains computational efficiency of ReLU

## Problems

- **Increased Complexity**: Learning α adds parameters, increasing computational cost
- **Not Zero-Centered**: Like ReLU and Leaky ReLU
- **Overfitting Risk**: More parameters can lead to overfitting on small datasets

## Implementation in Python

Since PReLU involves a learnable parameter, we'll simulate it with a fixed α for simplicity, but note that in a neural network, α would be updated via backpropagation.

```python
import numpy as np
import matplotlib.pyplot as plt

# PReLU function
def prelu(x, alpha=0.25):
    return np.where(x >= 0, x, alpha * x)

# Derivative of PReLU
def prelu_derivative(x, alpha=0.25):
    return np.where(x >= 0, 1, alpha)

# Generate input values
x = np.linspace(-10, 10, 100)
y = prelu(x)
y_deriv = prelu_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='PReLU Function', color='blue')
plt.plot(x, y_deriv, label='PReLU Derivative', color='red', linestyle='--')
plt.title('PReLU Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

### Code Explanation

- **PReLU Function**: Similar to Leaky ReLU, but α is typically learned. Here, we use α = 0.25
- **Derivative**: Same as Leaky ReLU, but in practice, α is updated via gradients
- **Visualization**: Similar to Leaky ReLU, but the negative slope depends on the learned α

## Key Differences from Leaky ReLU

| Aspect | Leaky ReLU | PReLU |
|--------|------------|-------|
| α parameter | Fixed (typically 0.01) | Learnable |
| Complexity | Lower | Higher |
| Adaptability | Static | Adaptive to data |
| Parameters | No additional parameters | One additional parameter per channel |

## When to Use PReLU

- Deep neural networks where gradient flow is critical
- When you have sufficient training data to learn the additional parameters
- Applications where the optimal negative slope varies across different parts of the network
- Image classification tasks with deep CNNs like ResNet
