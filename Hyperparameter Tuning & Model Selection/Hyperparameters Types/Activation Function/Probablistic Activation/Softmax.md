````markdown
## 1. Softmax Activation Function

### 📐 Mathematical Definition

The **Softmax** function transforms a vector of real-valued inputs  
\(\mathbf{x} = [x_1, x_2, \dots, x_n]\)  
into a probability distribution over \(n\) classes:

\[
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
\]

Where:
- \(x_i\) is the \(i\)-th input (logit) in the vector \(\mathbf{x}\).
- The numerator exponentiates the input to ensure non-negativity.
- The denominator normalizes outputs to sum to 1.

---

### 📊 Mathematical Properties

- **Range**:  
  \((0, 1)\) for each output, with  
  \(\sum_{i=1}^n \text{Softmax}(x_i) = 1\)

- **Derivative (Jacobian Matrix)**:  
  For the \(i\)-th output and \(j\)-th input:

\[
\frac{\partial \text{Softmax}(x_i)}{\partial x_j} = 
\begin{cases} 
\text{Softmax}(x_i) \cdot (1 - \text{Softmax}(x_i)) & \text{if } i = j \\
-\text{Softmax}(x_i) \cdot \text{Softmax}(x_j) & \text{if } i \neq j 
\end{cases}
\]

- **Behavior**:
  - Large positive \(x_i\) → output near 1 (dominates).
  - Large negative \(x_i\) → output near 0.
  - Equal inputs → equal probabilities.

- **Numerical Stability**:  
  To prevent overflow, subtract the maximum input:
  
\[
\text{Softmax}(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j=1}^n e^{x_j - \max(\mathbf{x})}}
\]

---

### 🎯 Use Cases

- **Multi-Class Classification**  
  Converts logits to class probabilities in neural networks.

- **Attention Mechanisms**  
  Computes attention weights in transformers.

---

### ✅ Advantages

- Outputs sum to 1: interpretable as probabilities.
- Smooth, differentiable.
- Standard for classification and attention.

---

### ⚠️ Problems

- Non-sparse outputs: assigns non-zero probability to all classes.
- Computationally expensive for large inputs.
- Sensitive to outlier logits.

---

### 💻 Implementation in Python

```python
import numpy as np
import matplotlib.pyplot as plt

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # numerical stability
    return exp_x / np.sum(exp_x)

# Derivative of Softmax (Jacobian for a single output)
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

# Example usage
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
````

---

### 📊 Explanation of Code

* **Softmax Function**: Computes normalized exponentials with stability correction.
* **Derivative Function**: Returns a row from the Jacobian, indicating sensitivity of each output to changes in inputs.
* **Visualization**: Shows class probability distribution for a sample input.

---

### 📌 Summary Table

| Property     | Value/Description                             |
| ------------ | --------------------------------------------- |
| Type         | Activation Function                           |
| Output Range | (0, 1), sums to 1                             |
| Use Case     | Multi-class classification, attention weights |
| Smoothness   | Fully differentiable                          |
| Limitation   | Non-sparse outputs, sensitive to large logits |
| Common in    | CNNs, RNNs, Transformers                      |

```
```
