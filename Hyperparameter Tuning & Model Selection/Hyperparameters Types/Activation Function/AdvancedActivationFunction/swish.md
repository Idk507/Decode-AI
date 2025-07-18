## 1. Swish Activation Function

### Mathematical Definition
The Swish activation function, proposed by Google Brain, is defined as:

\[
\text{Swish}(x) = x \cdot \sigma(\beta x)
\]

Where:
- \(\sigma(z) = \frac{1}{1 + e^{-z}}\) is the sigmoid function.
- \(\beta\) is a learnable parameter or fixed constant (often set to 1).
- When \(\beta = 1\), Swish simplifies to \(x \cdot \sigma(x)\).

### Mathematical Properties
- **Range**: \((-\infty, \infty)\), as the sigmoid \(\sigma(\beta x) \in (0, 1)\) scales \(x\), which can be positive or negative.
- **Derivative**:
  \[
  \frac{d}{dx} \text{Swish}(x) = \sigma(\beta x) + \beta x \cdot \sigma(\beta x) \cdot (1 - \sigma(\beta x))
  \]
  This uses the derivative of the sigmoid function: \(\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))\).
- **Behavior**:
  - For large positive \(x\), \(\sigma(\beta x) \to 1\), so \(\text{Swish}(x) \approx x\).
  - For large negative \(x\), \(\sigma(\beta x) \to 0\), so \(\text{Swish}(x) \approx 0\), but allows small negative outputs.
  - At \(x = 0\), \(\text{Swish}(0) = 0 \cdot \sigma(0) = 0 \cdot 0.5 = 0\).
- **Smoothness**: Swish is smooth and differentiable everywhere, unlike ReLU.
- **Non-Monotonic**: Unlike ReLU or sigmoid, Swish is not strictly increasing, as it allows negative outputs for negative inputs.

### Use Case
- **Deep Networks**: Outperforms ReLU in some deep architectures, such as convolutional neural networks (CNNs) and deep feedforward networks.
- Example: Used in Google’s Inception models and other large-scale image classification tasks.

### Advantages
- **Improved Performance**: Empirically outperforms ReLU in many deep learning tasks due to its smooth, non-linear behavior.
- **Smooth Gradients**: Avoids the non-differentiability of ReLU at \(x = 0\).
- **Learnable \(\beta\)**: When \(\beta\) is trainable, Swish adapts to the data, potentially improving performance.
- **Negative Outputs**: Allows small negative activations, reducing the dying neuron problem.

### Problems
- **Computationally Expensive**: Involves the sigmoid function, which requires computing exponentials.
- **Hyperparameter \(\beta\)**: When \(\beta\) is fixed, it may not be optimal for all tasks; when learnable, it adds parameters.
- **Not Zero-Centered**: Outputs are not symmetric around 0, which can affect gradient flow.

### Implementation in Python
Let’s implement Swish with a fixed \(\beta = 1\) and its derivative, then visualize it.

```python
import numpy as np
import matplotlib.pyplot as plt

# Swish function
def swish(x, beta=1.0):
    return x * (1 / (1 + np.exp(-beta * x)))

# Derivative of Swish
def swish_derivative(x, beta=1.0):
    sigmoid = 1 / (1 + np.exp(-beta * x))
    return sigmoid + beta * x * sigmoid * (1 - sigmoid)

# Generate input values
x = np.linspace(-10, 10, 100)
y = swish(x)
y_deriv = swish_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Swish Function', color='blue')
plt.plot(x, y_deriv, label='Swish Derivative', color='red', linestyle='--')
plt.title('Swish Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

**Explanation of Code**:
- **Swish Function**: Computes \(x \cdot \sigma(\beta x)\) with \(\beta = 1\).
- **Derivative**: Uses the product rule to compute \(\sigma(\beta x) + \beta x \cdot \sigma(\beta x) \cdot (1 - \sigma(\beta x))\).
- **Visualization**: The plot shows Swish resembling ReLU for positive inputs but allowing small negative outputs, with a smooth derivative that avoids vanishing gradients for moderate inputs.

---
