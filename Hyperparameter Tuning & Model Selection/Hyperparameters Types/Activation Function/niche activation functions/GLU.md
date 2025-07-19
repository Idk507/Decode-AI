
---

## 🔷 3. Gated Linear Unit (GLU) Activation Function

---

### 📐 Mathematical Definition

The **Gated Linear Unit (GLU)** applies a sigmoid gate to modulate its input:

$$
\text{GLU}(x) = x \cdot \sigma(Wx + b)
$$

Where:

* $x \in \mathbb{R}^d$ : input vector.
* $W \in \mathbb{R}^{d \times d}$, $b \in \mathbb{R}^d$ : learnable weights and biases.
* $\sigma(z) = \frac{1}{1 + e^{-z}}$ : sigmoid function.

---

### 📊 Mathematical Properties

| Property              | Description                                                                                                                |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Range**             | $(-\infty, \infty)$, since $\sigma(Wx + b) \in (0, 1)$ scales $x$, which is unbounded.                                     |
| **Differentiability** | Fully differentiable, smooth everywhere.                                                                                   |
| **Derivative**        | From the product rule:                                                                                                     |
|                       | $\displaystyle \frac{\partial \text{GLU}(x)}{\partial x_i} = \sigma(Wx + b)_i + x_i \cdot \sigma'(Wx + b)_i \cdot W_{i,:}$ |
| **Gating Behavior**   | Acts as a soft gate, controlling how much of each input dimension passes through.                                          |

* If $\sigma(Wx + b) \approx 1$, GLU behaves like identity: $\text{GLU}(x) \approx x$.
* If $\sigma(Wx + b) \approx 0$, GLU suppresses input: $\text{GLU}(x) \approx 0$.

---

### 🛠 Use Case

* **Sequence Modeling & Transformers**:

  * Widely used in models like **Transformer-XL**.
  * Enhances feature selection via gating in NLP tasks (machine translation, language modeling).
* **Gated Mechanisms**:

  * Similar in concept to **LSTM gates** or **GRU**, but simpler and applied in feed-forward contexts.

---

### ✅ Advantages

| Advantage             | Explanation                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| **Learnable Gating**  | Gate learns to suppress or amplify inputs dynamically during training.      |
| **Context-Sensitive** | Gate depends on input via affine transformation (context-aware modulation). |
| **Smooth Gradients**  | Derivative is continuous; avoids sharp transitions.                         |

---

### ⚠️ Problems

| Limitation                   | Explanation                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------- |
| **Extra Parameters**         | Requires weights and biases for gate computation.                               |
| **More Expensive than ReLU** | Matrix multiplications + sigmoid function are costlier than simple activations. |
| **Output Not Zero-Centered** | Like sigmoid, outputs can bias gradients in optimization.                       |

---

### 💻 Python Implementation

Example for scalar input using 1D affine gate:

```python
import numpy as np
import matplotlib.pyplot as plt

# GLU activation
def glu(x, w=1.0, b=0.0):
    sigmoid = 1 / (1 + np.exp(-(w * x + b)))
    return x * sigmoid

# Derivative of GLU
def glu_derivative(x, w=1.0, b=0.0):
    sigmoid = 1 / (1 + np.exp(-(w * x + b)))
    sigmoid_deriv = sigmoid * (1 - sigmoid)
    return sigmoid + x * sigmoid_deriv * w

# Generate input values
x = np.linspace(-10, 10, 100)
y = glu(x)
y_deriv = glu_derivative(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='GLU Function', color='blue')
plt.plot(x, y_deriv, label='GLU Derivative', color='red', linestyle='--')
plt.title('GLU Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 📊 Interpretation of Plot

* **GLU Function**:

  * Looks like a **gated identity** function.
  * Gradually allows $x$ to pass as sigmoid gate opens.

* **GLU Derivative**:

  * Peaks where sigmoid’s derivative is maximal.
  * Always smooth and non-zero.

---

### 📌 Summary

* **GLU** introduces **dynamic, learnable gates** that modulate activations based on input features.
* Suitable for tasks where **adaptive feature selection** is important.
* Common in **advanced sequence models** and architectures needing soft, context-aware gating.

---

