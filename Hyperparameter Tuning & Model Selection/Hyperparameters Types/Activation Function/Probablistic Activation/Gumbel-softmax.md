
---

## 🔷 3. Gumbel-Softmax Activation Function

---

### 📐 Mathematical Definition

**Gumbel-Softmax** approximates **categorical sampling** in a **differentiable** way. Given logits $\mathbf{x} = [x_1, \dots, x_n]$:

$$
\text{Gumbel-Softmax}(x_i) = \frac{e^{(x_i + g_i)/\tau}}{\sum_{j=1}^n e^{(x_j + g_j)/\tau}}
$$

Where:

* $g_i \sim \text{Gumbel}(0, 1)$
* $g_i = -\log(-\log(u_i))$ with $u_i \sim \text{Uniform}(0, 1)$
* $\tau > 0$ is the **temperature** controlling distribution sharpness:

  * $\tau \to 0$: Approximates **one-hot encoding**.
  * $\tau \to \infty$: Becomes uniform.

---

### 🎲 Gumbel Noise

Gumbel noise simulates sampling from a categorical distribution by perturbing logits:

* Ensures **argmax approximation** in a differentiable manner.
* Adds stochasticity during training.

---

### 📊 Mathematical Properties

| Property                           | Description                                                          |
| ---------------------------------- | -------------------------------------------------------------------- |
| **Range**                          | $[0, 1]$ for each output, summing to 1.                              |
| **Differentiability**              | Fully differentiable w\.r.t. $\mathbf{x}$, enabling backpropagation. |
| **Temperature Parameter ($\tau$)** | Controls smoothness vs. discreteness.                                |
| **Noise Source**                   | Gumbel distribution introduces randomness.                           |

---

**Behavior Based on Temperature $\tau$:**

* **Low $\tau$ (e.g., 0.1):**

  * Output approximates a **one-hot vector**.
  * Sharp, discrete-like sampling.

* **High $\tau$ (e.g., 5.0):**

  * Output resembles **Softmax**.
  * Smooth distribution over classes.

---

### 📈 Use Case

* **Differentiable Discrete Sampling**:

  * Replace non-differentiable **argmax** during training.
  * Learn **discrete latent variables** via backpropagation.

* **Applications**:

  * Variational Autoencoders (VAEs) with discrete latent spaces.
  * Generative models needing differentiable categorical choices.
  * Reinforcement learning policies requiring discrete sampling.

Example:

* **Text generation** with discrete token outputs.
* **Image synthesis** with discrete latent control.

---

### ✅ Advantages

| Advantage                   | Reason                                                                       |
| --------------------------- | ---------------------------------------------------------------------------- |
| **Differentiable Sampling** | Enables training models with discrete latent variables using gradients.      |
| **Stochastic Exploration**  | Gumbel noise promotes exploration during optimization.                       |
| **Temperature Control**     | $\tau$ offers flexibility between smooth probabilities and sharp selections. |

---

### ⚠️ Problems

| Limitation                  | Explanation                                              |
| --------------------------- | -------------------------------------------------------- |
| **Temperature Sensitivity** | Incorrect $\tau$ tuning can harm model performance.      |
| **Stochastic Output**       | Adds randomness, possibly destabilizing early training.  |
| **Computational Overhead**  | Requires sampling Gumbel noise and Softmax computations. |

---

### 💻 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Gumbel-Softmax function
def gumbel_softmax(x, tau=0.5):
    # Sample Gumbel noise
    u = np.random.uniform(0, 1, size=x.shape)
    g = -np.log(-np.log(u + 1e-20) + 1e-20)
    
    # Apply Gumbel-Softmax
    y = np.exp((x + g) / tau)
    return y / np.sum(y)

# Example usage
x = np.array([1.0, 2.0, 3.0, 4.0])
y = gumbel_softmax(x, tau=0.5)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(len(x)), y, color='blue', alpha=0.7)
plt.title('Gumbel-Softmax Activation Function Output (τ=0.5)')
plt.xlabel('Class Index')
plt.ylabel('Probability')
plt.xticks(range(len(x)), [f'Class {i+1}' for i in range(len(x))])
plt.grid(True)
plt.show()
```

---

### 📊 Code Explanation

* **Gumbel Noise Sampling**: Adds randomness to logits.
* **Temperature Scaling**: Controls sharpness of output.
* **Softmax Operation**: Converts perturbed logits to probabilities.
* **Visualization**: Bar chart shows stochastic, soft probability distribution.

---

### 🔍 Comparison: Gumbel-Softmax vs. Softmax

| Feature                | Gumbel-Softmax                           | Softmax                   |
| ---------------------- | ---------------------------------------- | ------------------------- |
| Output Type            | Smooth (high $\tau$), Sharp (low $\tau$) | Always smooth             |
| Sampling               | Stochastic via Gumbel noise              | Deterministic             |
| Differentiability      | Yes                                      | Yes                       |
| Discrete Approximation | Possible (low $\tau$)                    | No (outputs always dense) |
| Use in Discrete VAEs   | Common                                   | Not suitable              |

---

### 🚀 Summary

* **Gumbel-Softmax** bridges the gap between **discrete sampling** and **gradient-based optimization**.
* Stochastic yet differentiable, enabling neural networks to **sample categories during training**.
* Widely applied in **discrete latent VAEs**, **reinforcement learning**, and **generative models**.

---

