
---

## 🔷 4. Snake Activation Function

---

### 📐 Mathematical Definition

The **Snake activation function** blends linearity with periodic oscillations, ideal for modeling periodic structures:

$$
\text{Snake}(x) = x + \frac{1 - \cos(2x)}{2}
$$

Using trigonometric identity:

$$
\frac{1 - \cos(2x)}{2} = \sin^2(x)
$$

Therefore:

$$
\text{Snake}(x) = x + \sin^2(x)
$$

---

### 📊 Mathematical Properties

| Property        | Description                                                                              |
| --------------- | ---------------------------------------------------------------------------------------- |
| **Range**       | $(-\infty, \infty)$. The $\sin^2(x)$ term oscillates between 0 and 1.                    |
| **Derivative**  | $\displaystyle \frac{d}{dx} \text{Snake}(x) = 1 + \sin(2x)$.                             |
| **Periodicity** | Introduces oscillations with period $\pi$.                                               |
| **Behavior**    | - Acts as a periodic modulation over linearity.<br>- Oscillates around the line $y = x$. |

* At $x = 0$:

  $$
  \text{Snake}(0) = 0
  $$
* At large $|x|$, oscillations persist over a linear trend.

---

### 🛠 Use Case

* **Periodic Data Modeling**:

  * Time-series data with **seasonal cycles**.
  * Physical systems with **wave-like behaviors**.
  * Tasks where representing **periodic signals** directly within the activation function is beneficial.

Example domains:

* Seasonal sales prediction.
* Signal processing.
* Physics-based modeling.

---

### ✅ Advantages

| Advantage                     | Explanation                                                     |
| ----------------------------- | --------------------------------------------------------------- |
| **Periodic Non-Linearity**    | Directly incorporates periodicity, unlike standard activations. |
| **Smooth Differentiability**  | Continuous derivative simplifies backpropagation.               |
| **Handles Unbounded Outputs** | Suitable for tasks requiring large-range regression outputs.    |

---

### ⚠️ Limitations

| Limitation                   | Explanation                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| **Niche Usage**              | Less effective outside periodic data contexts.                 |
| **Computational Overhead**   | Trigonometric calculations are more expensive than linear ops. |
| **Non-Zero Centered Output** | Output biased upwards due to additive $\sin^2(x)$.             |

---

### 💻 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Snake activation function
def snake(x):
    return x + np.sin(x)**2

# Derivative of Snake
def snake_derivative(x):
    return 1 + np.sin(2 * x)

# Generate input
x = np.linspace(-10, 10, 100)
y = snake(x)
y_deriv = snake_derivative(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Snake Function', color='blue')
plt.plot(x, y_deriv, label='Snake Derivative', color='red', linestyle='--')
plt.title('Snake Activation Function and Its Derivative')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 📊 Interpretation

* **Function Plot**:

  * Linear trend overlaid with oscillations.
* **Derivative Plot**:

  * Oscillates around 1, as $\sin(2x) \in [-1, 1]$.

---

### 📌 Summary

| Feature           | Snake Activation                                                   |
| ----------------- | ------------------------------------------------------------------ |
| **Best For**      | Modeling periodic features in regression and time-series problems. |
| **Core Behavior** | Adds bounded, periodic non-linearity to linear activations.        |
| **Trade-Off**     | Gained periodic modeling power at higher computational cost.       |

---
