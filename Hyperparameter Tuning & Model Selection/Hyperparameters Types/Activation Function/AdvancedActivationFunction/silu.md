
---

# SiLU (Sigmoid-Weighted Linear Unit) Activation Function

## Mathematical Definition

SiLU is a special case of the **Swish** function with $\beta = 1$:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

Where:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

---

## Mathematical Properties

* **Range**: $(-\infty, \infty)$

* **Derivative**:

$$
\frac{d}{dx} \text{SiLU}(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))
$$

* **Behavior**:

  * For large positive $x$: behaves like ReLU ($\text{SiLU}(x) \approx x$)
  * For large negative $x$: allows small negative outputs
  * Smooth and differentiable everywhere

* **Smoothness**: Fully differentiable, like Swish.

---

## Use Cases

* **Efficient Architectures**:

  * Used in models like **EfficientNet** and other mobile-optimized neural networks.
  * Suitable for **real-time inference** and edge deployments.

---

## Advantages

* **Simpler than Swish**: No learnable $\beta$ parameter.
* **Smooth Gradients**: Avoids issues like dying neurons.
* **Outperforms ReLU**: In certain tasks, particularly in computer vision.

---

## Problems

* **Computationally Expensive**: Involves computing the sigmoid function.
* **Not Zero-Centered**: Like Swish, outputs skew positive.
* **Fixed Behavior**: No learnable parameter limits flexibility.

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# SiLU activation function
def silu(x):
    return x * (1 / (1 + np.exp(-x)))

# Derivative of SiLU
def silu_derivative(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid + x * sigmoid * (1 - sigmoid)

# Generate input values
x = np.linspace(-10, 10, 100)
y = silu(x)
y_deriv = silu_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='SiLU Function', color='blue')
plt.plot(x, y_deriv, label='SiLU Derivative', color='red', linestyle='--')
plt.title('SiLU Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Explanation of Code

* **SiLU Function**: Computes $x \cdot \sigma(x)$, equivalent to Swish with $\beta = 1$.
* **Derivative**: Smooth derivative ensures stable optimization.
* **Visualization**:

  * Illustrates SiLU’s smooth transition from negative to positive inputs.
  * Derivative shows gradual, non-zero gradients across input range.

---


