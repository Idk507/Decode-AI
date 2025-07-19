

---

# GELU (Gaussian Error Linear Unit) Activation Function

## Mathematical Definition

The **GELU** function, commonly used in transformer architectures, is defined as:

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

Where:

* $\Phi(x) = P(Z \leq x)$ is the **cumulative distribution function (CDF)** of the standard normal distribution $(Z \sim \mathcal{N}(0, 1))$.

### Approximations:

* Using sigmoid:

$$
\text{GELU}(x) \approx x \cdot \sigma(1.702 x)
$$

* Using tanh:

$$
\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
$$

---

## Mathematical Properties

* **Range**: $(-\infty, \infty)$, since $\Phi(x) \in (0, 1)$ and $x$ spans all real numbers.

* **Derivative** (approximate, using sigmoid form):

$$
\frac{d}{dx} \text{GELU}(x) \approx \sigma(1.702 x) + 1.702 x \cdot \sigma(1.702 x) \cdot (1 - \sigma(1.702 x))
$$

* **Behavior**:

  * For large positive $x$: $\Phi(x) \to 1$, so $\text{GELU}(x) \approx x$.
  * For large negative $x$: $\Phi(x) \to 0$, so $\text{GELU}(x) \approx 0$.
  * **Smooth** and **differentiable** everywhere.

* **Probabilistic Interpretation**: GELU weights $x$ by the probability that a normally distributed random variable falls below $x$.

---

## Use Cases

* **Transformers**: Standard activation function in models like **BERT**, **GPT**, **T5**, etc.
* Applied in tasks like:

  * Sentiment analysis
  * Machine translation
  * Question answering

---

## Advantages

* **Smooth & Probabilistic**: Provides a theoretically motivated non-linearity with smooth gradients.
* **Empirical Success**: Demonstrates superior performance in NLP tasks compared to ReLU.
* **Non-Monotonic**: Allows small negative outputs, reducing the dying neuron issue.

---

## Drawbacks

* **Computationally Expensive**: Exact GELU requires evaluating the Gaussian CDF.
* **Approximation Required**: Tanh or sigmoid approximations are typically used for practical efficiency.
* **Not Zero-Centered**: Output values are not centered around zero.
* **Less Intuitive**: Compared to simpler activations like ReLU.

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# GELU function (tanh approximation)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Derivative of GELU (approximate, using numerical differentiation)
def gelu_derivative(x, h=1e-5):
    return (gelu(x + h) - gelu(x - h)) / (2 * h)

# Generate input values
x = np.linspace(-10, 10, 100)
y = gelu(x)
y_deriv = gelu_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='GELU Function', color='blue')
plt.plot(x, y_deriv, label='GELU Derivative', color='red', linestyle='--')
plt.title('GELU Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Explanation of Code

* **GELU Function**: Implemented using the **tanh approximation** for computational efficiency.
* **Derivative**: Approximated using **numerical differentiation** due to the complexity of the analytical derivative.
* **Visualization**:

  * Shows GELU behaving similarly to ReLU for positive inputs.
  * Allows small negative outputs.
  * Derivative remains smooth and continuous across all input values.

---


