
---

## 1. Softmax Activation Function

---

### 📐 Mathematical Definition

The **Softmax function** transforms a vector of real-valued inputs $\mathbf{x} = [x_1, x_2, \dots, x_n]$ into a probability distribution:

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

Where:

* $x_i$ is the $i$-th input (logit).
* $e^{x_i}$ ensures non-negative outputs.
* The denominator normalizes outputs to sum to 1.

---

### 📊 Mathematical Properties



<img width="851" height="489" alt="image" src="https://github.com/user-attachments/assets/5db64226-12c8-436a-abf6-6291ce46cd99" />


---

### ⚙️ Use Cases

* **Multi-Class Classification**:

  * Common in the output layer of neural networks.
  * Example: Classifying an image (e.g., dog, cat, bird).

* **Attention Mechanisms**:

  * Used in transformers to compute attention weights.

---

### ✅ Advantages

| Advantage                       | Description                                         |
| ------------------------------- | --------------------------------------------------- |
| **Interpretable Probabilities** | Outputs sum to 1, representing class probabilities. |
| **Smooth and Differentiable**   | Supports gradient-based optimization.               |
| **Widely Used**                 | Standard choice for classification and attention.   |

---

### ⚠️ Problems

| Problem                     | Explanation                                                  |
| --------------------------- | ------------------------------------------------------------ |
| **Non-Sparse Outputs**      | Assigns non-zero probability to all classes.                 |
| **Computational Cost**      | Involves exponentials and summations (costly for large $n$). |
| **Sensitivity to Outliers** | Large inputs can skew the distribution heavily.              |

---

### 💻 Implementation in Python

```python
import numpy as np
import matplotlib.pyplot as plt

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

# Derivative of Softmax (Jacobian row for a single element)
def softmax_derivative(x, i):
    s = softmax(x)
    jacobian = np.zeros((len(x), len(x)))
    for j in range(len(x)):
        for k in range(len(x)):
            if j == k:
                jacobian[j, k] = s[j] * (1 - s[j])
            else:
                jacobian[j, k] = -s[j] * s[k]
    return jacobian[i]

# Sample input
x = np.array([1.0, 2.0, 3.0, 4.0])
y = softmax(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(len(x)), y, color='blue', alpha=0.7)
plt.title('Softmax Activation Function Output')
plt.xlabel('Class Index')
plt.ylabel('Probability')
plt.xticks(range(len(x)), [f'Class {i+1}' for i in range(len(x))])
plt.grid(True)
plt.show()
```

---

### 📈 Code Explanation

* **Softmax Function**:

  * Computes the probability distribution using exponentials.
  * Ensures numerical stability by subtracting the max input.

* **Derivative**:

  * Computes the Jacobian matrix row for a selected output.

* **Visualization**:

  * Shows the probability distribution as a bar chart.
  * Demonstrates how Softmax highlights higher inputs.

---


