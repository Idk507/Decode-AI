
---

# Swish Activation Function

## Mathematical Definition

The **Swish** activation function, proposed by Google Brain, is defined as:

$$
\text{Swish}(x) = x \cdot \sigma(\beta x)
$$

Where:

* $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the **sigmoid function**.
* $\beta$ is a **learnable parameter** or fixed constant (often set to 1).
* When $\beta = 1$, Swish simplifies to:

$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

---

## Mathematical Properties

* **Range**: $(-\infty, \infty)$, as $\sigma(\beta x) \in (0, 1)$ and $x$ can be positive or negative.

* **Derivative**:

$$
\frac{d}{dx} \text{Swish}(x) = \sigma(\beta x) + \beta x \cdot \sigma(\beta x) \cdot (1 - \sigma(\beta x))
$$

(using $\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$).

* **Behavior**:

  * For large positive $x$, $\sigma(\beta x) \to 1$, so $\text{Swish}(x) \approx x$.
  * For large negative $x$, $\sigma(\beta x) \to 0$, so $\text{Swish}(x) \approx 0$, but allows small negative outputs.
  * At $x = 0$, $\text{Swish}(0) = 0$.

* **Smoothness**: Smooth and differentiable everywhere, unlike ReLU.

* **Non-Monotonic**: Allows small negative outputs; not strictly increasing.

---

## Use Cases

* **Deep Neural Networks**: Performs better than ReLU in some architectures like CNNs and deep feedforward networks.
* Example: Used in Google's Inception models and large-scale image classification tasks.

---

## Advantages

* **Improved Performance**: Outperforms ReLU in many tasks due to smooth non-linearity.
* **Smooth Gradients**: Avoids non-differentiability issues present in ReLU.
* **Learnable $\beta$**: Allows adaptability during training.
* **Negative Outputs**: Reduces the dying neuron problem.

---

## Drawbacks

* **Computationally Expensive**: Requires computing exponentials (sigmoid).
* **Hyperparameter $\beta$**: Adds complexity if trainable.
* **Not Zero-Centered**: Can affect gradient flow during optimization.

---

## Python Implementation

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

---

## Explanation of Code

* **Swish Function**: Computes $x \cdot \sigma(\beta x)$ with $\beta = 1$.
* **Derivative**: Derived using the product rule and derivative of sigmoid.
* **Visualization**:

  * Swish resembles ReLU for large positive inputs.
  * Allows small negative outputs.
  * Smooth derivative helps prevent vanishing gradients.

---

