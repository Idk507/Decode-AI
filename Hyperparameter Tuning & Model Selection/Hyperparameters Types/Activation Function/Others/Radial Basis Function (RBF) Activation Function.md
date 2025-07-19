
---

## 🔷 3. Radial Basis Function (RBF) Activation Function

---

### 📐 Mathematical Definition

For scalar input:

$$
\text{RBF}(x) = e^{-\frac{(x - c)^2}{2\sigma^2}}
$$

Where:

* $x$: Input value.
* $c$: Center of the Gaussian.
* $\sigma$: Width parameter controlling spread.

For vector inputs:

$$
\text{RBF}(x) = e^{-\frac{\|x - c\|^2}{2\sigma^2}}
$$

---

### 📊 Mathematical Properties

| Property       | Description                                                                     |
| -------------- | ------------------------------------------------------------------------------- |
| **Range**      | $(0, 1]$. Maximum at $x = c$.                                                   |
| **Derivative** | $-\frac{(x - c)}{\sigma^2} \cdot \text{RBF}(x)$.                                |
| **Response**   | Strong near $x = c$; weak for distant inputs.                                   |
| **Locality**   | **Localized activation** — outputs significant values only near the center $c$. |
| **Smoothness** | Infinitely differentiable (smooth).                                             |
| **Shape**      | Bell curve (Gaussian).                                                          |

---

### 🛠 Use Case

| Application                | Explanation                                                         |
| -------------------------- | ------------------------------------------------------------------- |
| **RBF Networks**           | Specialized networks for interpolation, clustering, classification. |
| **Kernel Methods**         | RBF is related to Gaussian kernels used in support vector machines. |
| **Function Approximation** | Can approximate any continuous function (universal approximator).   |

* **Example**: Time-series prediction, pattern recognition, spatial interpolation.

---

### ✅ Advantages

* **Localized Non-Linearity**: Activations depend on distance from centers.
* **Smooth Gradient**: Differentiable everywhere, aiding optimization.
* **Interpretability**: Centers can be interpreted as prototype patterns.
* **Strong Approximation Capability** (when combined properly in RBF networks).

---

### ⚠️ Limitations

| Limitation                         | Reason                                                                                  |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| **Center Tuning**                  | Requires careful setting or learning of $c$ and $\sigma$.                               |
| **Not Suitable for Standard MLPs** | Rarely used in feedforward or convolutional architectures.                              |
| **Computational Cost**             | Exponential and norm computations are expensive, especially in high-dimensional spaces. |

---

### 💻 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# RBF Activation Function
def rbf(x, c=0.0, sigma=1.0):
    return np.exp(-((x - c)**2) / (2 * sigma**2))

# Derivative of RBF
def rbf_derivative(x, c=0.0, sigma=1.0):
    return -((x - c) / sigma**2) * rbf(x, c, sigma)

# Generate input values
x = np.linspace(-10, 10, 200)
y = rbf(x)
y_deriv = rbf_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='RBF Function', color='blue')
plt.plot(x, y_deriv, label='RBF Derivative', color='red', linestyle='--')
plt.title('RBF Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 📊 Visualization Insights

* **RBF Function**: Symmetric Gaussian bell curve centered at $c = 0$.
* **Derivative**: Antisymmetric curve — gradient peaks around the center, zero at the peak itself.

---

### 📌 Summary Table

| Feature                | RBF Activation Function                                            |
| ---------------------- | ------------------------------------------------------------------ |
| **Purpose**            | Localized activation around center $c$.                            |
| **Best For**           | RBF networks, interpolation, distance-based tasks.                 |
| **Gradient Profile**   | Smooth, localized gradients.                                       |
| **Hidden Layer Use**   | Yes — core to RBF networks.                                        |
| **Computational Cost** | High due to exponential evaluations.                               |
| **Alternative**        | Gaussian kernels (in kernel methods), other localized activations. |

---

