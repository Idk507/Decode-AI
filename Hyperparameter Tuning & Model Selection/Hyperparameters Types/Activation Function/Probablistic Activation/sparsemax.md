
---

## 🔷 2. Sparsemax Activation Function

---

### 📐 Mathematical Definition

The **Sparsemax** function projects an input vector $\mathbf{x} \in \mathbb{R}^n$ onto the **probability simplex**, producing a **sparse probability distribution**.

It solves the following optimization problem:

$$
\text{Sparsemax}(\mathbf{x}) = \arg\min_{\mathbf{p} \in \Delta^{n-1}} \|\mathbf{p} - \mathbf{x}\|_2^2
$$

Where:

* $\Delta^{n-1} = \{\mathbf{p} \in \mathbb{R}^n : \mathbf{p} \geq 0, \sum_{i=1}^n p_i = 1\}$
  (the $(n-1)$-dimensional probability simplex).

---

**Closed-form Solution:**

$$
\text{Sparsemax}(x_i) = \max(0, x_i - \tau)
$$

Where:

* $\tau$ is a **threshold** calculated such that:

$$
\sum_i \text{Sparsemax}(x_i) = 1
$$

$\tau$ is found by:

1. Sorting inputs $x_{(1)} \geq x_{(2)} \geq \dots \geq x_{(n)}$
2. Finding the largest $k$ satisfying:

$$
x_{(k)} > \frac{\sum_{j=1}^k x_{(j)} - 1}{k}
$$

3. Then computing:

$$
\tau = \frac{\sum_{j=1}^k x_{(j)} - 1}{k}
$$

---

### 📊 Mathematical Properties

| Property             | Description                                      |
| -------------------- | ------------------------------------------------ |
| **Range**            | $[0, 1]$, summing to 1 (like Softmax)            |
| **Sparsity**         | Can output exact zeros (unlike Softmax)          |
| **Sum to One**       | Ensured by projection onto simplex               |
| **Piecewise Linear** | Outputs are piecewise linear functions of inputs |
| **Non-Negative**     | All outputs are non-negative                     |

---

**Jacobian (Derivative):**

The derivative of Sparsemax is sparse:

$$
\frac{\partial \text{Sparsemax}(x_i)}{\partial x_j} =
\begin{cases} 
1 - \frac{1}{k} & \text{if } i = j \text{ and } \text{Sparsemax}(x_i) > 0 \\
-\frac{1}{k} & \text{if } i \neq j \text{ and } \text{Sparsemax}(x_i), \text{Sparsemax}(x_j) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

Where $k$ is the number of non-zero outputs.

---

### 📈 Behavior

* **Sparse Output**: Focuses on top elements by assigning non-zero probabilities only to dominant classes.
* **Non-Smooth Transitions**: Unlike Softmax, Sparsemax’s output changes abruptly when elements cross the threshold.
* **Probability Distribution**: Like Softmax, outputs sum to 1, but some outputs are exact 0.

---

### 🛠️ Use Cases

* **Attention Mechanisms**: Improves interpretability by focusing attention on a few important tokens.
* **Multi-class Classification**: When sparsity in the output distribution is desired.
* **Text Summarization & Translation**: Helps models concentrate attention selectively.

---

### ✅ Advantages

| Advantage                      | Reason                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| **Sparse Outputs**             | Facilitates interpretability and efficiency by zeroing irrelevant probabilities.     |
| **Differentiable**             | Supports gradient-based optimization despite producing sparse results.               |
| **Less Sensitive to Outliers** | Unlike Softmax, extreme input values don't dominate unless they cross the threshold. |

---

### ⚠️ Limitations

| Problem                       | Explanation                                                            |
| ----------------------------- | ---------------------------------------------------------------------- |
| **More Complex Than Softmax** | Requires sorting and threshold computation.                            |
| **Less Popular**              | Rarely implemented in standard libraries.                              |
| **Hard Thresholding**         | Might discard mildly relevant classes by setting them exactly to zero. |

---

### 💻 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Sparsemax function
def sparsemax(x):
    z = np.sort(x)[::-1]                          # Sort descending
    z_cumsum = np.cumsum(z)
    k = np.arange(1, len(x) + 1)
    
    # Find k satisfying condition
    z_check = z - (z_cumsum - 1) / k
    k_valid = np.where(z_check > 0)[0]
    k_max = k_valid[-1] + 1                        # Convert to 1-based index
    
    # Compute threshold tau
    tau = (np.sum(z[:k_max]) - 1) / k_max
    
    # Compute projection
    return np.maximum(0, x - tau)

# Example usage
x = np.array([1.0, 2.0, 3.0, 4.0])
y = sparsemax(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(len(x)), y, color='blue', alpha=0.7)
plt.title('Sparsemax Activation Function Output')
plt.xlabel('Class Index')
plt.ylabel('Probability')
plt.xticks(range(len(x)), [f'Class {i+1}' for i in range(len(x))])
plt.grid(True)
plt.show()
```

---

### 📊 Explanation of Code

* **Sorting**: Inputs sorted to identify active components.
* **Threshold ($\tau$)**: Computed to determine cutoff between zeroed and active outputs.
* **Projection**: Using $\max(0, x_i - \tau)$, outputs sparse probabilities.
* **Visualization**: Displays how Sparsemax allocates probability mass only to dominant classes.

---

### 🔍 Sparsemax vs. Softmax

| Feature          | Sparsemax                     | Softmax                            |
| ---------------- | ----------------------------- | ---------------------------------- |
| Output           | Sparse (exact zeros)          | Dense (non-zero for all classes)   |
| Interpretability | High (focuses on few classes) | Lower (all classes receive weight) |
| Smoothness       | Piecewise linear (non-smooth) | Smooth                             |
| Dominance        | Focuses on top classes        | Spreads across all                 |

---

### 🚀 Summary

* **Sparsemax** is a **sparse alternative** to Softmax, assigning **zero probabilities** to less significant inputs.
* Produces **sparse, interpretable, and differentiable** outputs.
* Valuable in attention mechanisms where focusing sharply on key elements is important.

---
