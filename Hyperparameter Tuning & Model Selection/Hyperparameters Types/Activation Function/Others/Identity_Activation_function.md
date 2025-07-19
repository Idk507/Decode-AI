
---

## 🔷 2. Identity Activation Function

---

### 📐 Mathematical Definition

The **Identity Function** passes input directly as output:

$$
\text{Identity}(x) = x
$$

* Simple linear function.
* No transformation applied.

---

### 📊 Mathematical Properties

| Property                 | Description                                                     |
| ------------------------ | --------------------------------------------------------------- |
| **Range**                | $(-\infty, \infty)$ — output is unbounded.                      |
| **Derivative**           | Constant $1$.                                                   |
| **Non-linearity**        | **None** — linear function.                                     |
| **Continuity**           | Continuous and smooth.                                          |
| **Gradient Suitability** | Gradient is constant — avoids vanishing or exploding gradients. |

---

### 🛠 Use Case

| Application                     | Explanation                                                    |
| ------------------------------- | -------------------------------------------------------------- |
| **Output Layer for Regression** | Used when predicting continuous values (e.g., prices, scores). |
| **Linear Models**               | Equivalent to no activation — applies in simple linear models. |

* Example: Predicting house prices, stock prices, or sensor readings.

---

### ✅ Advantages

* **Simplicity**: Minimal computation — just passes input through.
* **Unbounded Output**: Suitable for tasks needing unrestricted continuous outputs.
* **Consistent Gradient**: Constant derivative ensures stable gradient flow.
* **No Risk of Vanishing/Exploding Gradients**: Unlike sigmoid or tanh in deep networks.

---

### ⚠️ Limitations

| Limitation                    | Reason                                                                              |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| **No Non-Linearity**          | Prevents learning complex patterns — unsuitable for hidden layers.                  |
| **Limited Expressiveness**    | Networks composed only of linear functions are equivalent to a single linear layer. |
| **No Feature Transformation** | Fails to induce non-linear feature spaces.                                          |

---

### 💻 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Identity Activation Function
def identity(x):
    return x

# Derivative of Identity
def identity_derivative(x):
    return np.ones_like(x)

# Generate input values
x = np.linspace(-10, 10, 100)
y = identity(x)
y_deriv = identity_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Identity Function', color='blue')
plt.plot(x, y_deriv, label='Identity Derivative', color='red', linestyle='--')
plt.title('Identity Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 📊 Visualization Insights

* **Identity Function**: Straight line, slope 1 — simple linear pass-through.
* **Derivative**: Constant at 1, reinforcing the uniform gradient property.

---

### 📌 Summary Table

| Feature                | Identity Function                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| **Purpose**            | Output raw continuous values (regression).                                                            |
| **Best For**           | Output layers in regression models.                                                                   |
| **Hidden Layer Use**   | Not recommended — lacks non-linearity.                                                                |
| **Computational Cost** | Minimal (just forwarding input).                                                                      |
| **Alternative**        | None needed for regression outputs; for hidden layers, use non-linear activations (ReLU, Tanh, etc.). |

---

