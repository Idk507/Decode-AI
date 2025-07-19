
---

## 🔷 1. Bent Identity Activation Function

---

### 📐 Mathematical Definition

The **Bent Identity** function introduces a slight non-linearity to a near-linear transformation:

$$
\text{BentId}(x) = \frac{\sqrt{x^2 + 1} - 1}{2} + x
$$

* Combines **non-linear bending** with a linear identity term.

---

### 📊 Mathematical Properties

| Property                | Description                                                                    |   |                |
| ----------------------- | ------------------------------------------------------------------------------ | - | -------------- |
| **Range**               | $(-\infty, \infty)$; unbounded on both sides.                                  |   |                |
| **Derivative**          | $\displaystyle \frac{d}{dx} \text{BentId}(x) = \frac{x}{2 \sqrt{x^2 + 1}} + 1$ |   |                |
| **Smoothness**          | Fully differentiable, continuous, and smooth everywhere.                       |   |                |
| **Asymptotic Behavior** | Approaches linear behavior as (                                                | x | ) grows large. |
| **Monotonicity**        | Strictly increasing (since derivative always > 0).                             |   |                |
| **Non-Linearity**       | Slight curvature near $x = 0$, linear for large (                              | x | ).             |

---

**Derivative Analysis**:

* At $x = 0$:

  $$
  \frac{d}{dx} \text{BentId}(0) = 1
  $$
* For large $x$:

  * Positive side: Derivative approaches $1.5$.
  * Negative side: Derivative approaches $0.5$.

---

### 📈 Behavior Overview

| Input Region    | Behavior Description                            |
| --------------- | ----------------------------------------------- |
| $x \to -\infty$ | Slightly scaled linear, approximates $0.5x$.    |
| $x \approx 0$   | Non-linear bend around origin.                  |
| $x \to +\infty$ | Slightly amplified linear, approximates $1.5x$. |

---

### 🚀 Use Case

* **Regression Tasks**: Smooth, differentiable outputs without saturation.
* Suitable when:

  * Continuous output prediction is required.
  * Gradients need to remain smooth and non-zero (avoiding ReLU's dead neurons).

Examples:

* Predicting house prices, stock values, or sensor readings in neural networks.

---

### ✅ Advantages

| Advantage                    | Reason                                                    |
| ---------------------------- | --------------------------------------------------------- |
| **Smooth Differentiability** | No sharp transitions, unlike ReLU or leaky ReLU.          |
| **Unbounded Outputs**        | Supports regression tasks naturally.                      |
| **No Dead Neurons**          | Always provides non-zero gradients.                       |
| **Slight Non-Linearity**     | Provides flexibility while retaining near-linear mapping. |

---

### ⚠️ Problems

| Limitation                   | Explanation                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------- |
| **Rare Usage**               | Not commonly found in standard libraries or pretrained models.                   |
| **Computational Overhead**   | Square root computation more expensive than ReLU's simple thresholding.          |
| **Non Zero-Centered Output** | Outputs skewed positive due to additive $x$, which can affect gradient symmetry. |

---

### 💻 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Bent Identity function
def bent_identity(x):
    return (np.sqrt(x**2 + 1) - 1) / 2 + x

# Derivative of Bent Identity
def bent_identity_derivative(x):
    return x / (2 * np.sqrt(x**2 + 1)) + 1

# Generate input values
x = np.linspace(-10, 10, 100)
y = bent_identity(x)
y_deriv = bent_identity_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Bent Identity Function', color='blue')
plt.plot(x, y_deriv, label='Bent Identity Derivative', color='red', linestyle='--')
plt.title('Bent Identity Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 📊 Code Explanation

* **Function**: Computes $\frac{\sqrt{x^2 + 1} - 1}{2} + x$.
* **Derivative**: Computes $\frac{x}{2 \sqrt{x^2 + 1}} + 1$.
* **Plot**:

  * Bent Identity curve is nearly linear but subtly curved around $x = 0$.
  * Derivative curve shows scaling effect:

    * > 1 for positive inputs.
    * <1 for negative inputs.

---

### 📌 Summary

* **Bent Identity**: A **rare**, **smooth**, slightly **non-linear activation** for regression.
* Advantages include avoiding vanishing gradients and maintaining differentiability.
* Trade-offs include computational cost and lack of adoption.

---

