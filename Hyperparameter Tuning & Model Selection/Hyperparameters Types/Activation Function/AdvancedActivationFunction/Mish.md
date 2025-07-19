
---

# Mish Activation Function

## Mathematical Definition

The **Mish** activation function, introduced for computer vision applications, is defined as:

$$
\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x))
$$

Where:

* $\text{softplus}(x) = \log(1 + e^x)$
* $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

---

## Mathematical Properties

* **Range**: $(-\infty, \infty)$, since $\tanh(\text{softplus}(x)) \in (-1, 1)$ and $x$ spans all real values.

* **Derivative**:

$$
\frac{d}{dx} \text{Mish}(x) = \tanh(\text{softplus}(x)) + x \cdot \text{sech}^2(\text{softplus}(x)) \cdot \sigma(x)
$$

Where:

* $\sigma(x) = \frac{1}{1 + e^{-x}}$ (sigmoid function)

* $\text{sech}(z) = \frac{1}{\cosh(z)}$

* **Behavior**:

  * For large positive $x$: $\text{Mish}(x) \approx x$
  * For large negative $x$: $\text{Mish}(x) \to 0$
  * Smooth and differentiable everywhere.

* **Non-Monotonic**: Allows negative outputs for negative inputs.

---

## Use Cases

* **Computer Vision Tasks**:

  * Used in models like **YOLOv4** for real-time object detection.
  * Suitable for tasks such as object detection, image segmentation, etc.

---

## Advantages

* **Improved Performance**: Outperforms ReLU and Swish in certain vision applications.
* **Smooth Gradients**: Combines the smooth properties of **softplus** and **tanh**.
* **Negative Outputs**: Reduces the dying neuron problem by allowing small negative activations.

---

## Problems

* **Computationally Expensive**: Requires evaluating exponentials, logarithms, and hyperbolic tangent functions.
* **Not Zero-Centered**: Outputs are not symmetric around zero.
* **Less Adoption**: Compared to ReLU or GELU, Mish is less studied and used.

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Mish function
def mish(x):
    softplus = np.log(1 + np.exp(x))
    return x * np.tanh(softplus)

# Derivative of Mish
def mish_derivative(x):
    softplus = np.log(1 + np.exp(x))
    sigmoid = 1 / (1 + np.exp(-x))
    return np.tanh(softplus) + x * (1 - np.tanh(softplus)**2) * sigmoid

# Generate input values
x = np.linspace(-10, 10, 100)
y = mish(x)
y_deriv = mish_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Mish Function', color='blue')
plt.plot(x, y_deriv, label='Mish Derivative', color='red', linestyle='--')
plt.title('Mish Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Explanation of Code

* **Mish Function**: Computes $x \cdot \tanh(\log(1 + e^x))$.
* **Derivative**: Uses the chain rule to compute the derivative involving **tanh**, **softplus**, and **sigmoid**.
* **Visualization**:

  * Shows Mish’s similarity to **Swish**, combining smooth non-linearity with ReLU-like behavior for positive inputs.
  * Allows controlled negative outputs.


