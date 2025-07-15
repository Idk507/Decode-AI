
---

## 🧠 What Are Quasi-Newton Methods?

### 🔄 Problem with Newton’s Method:

Newton’s Method uses the **Hessian Matrix** (second-order derivative), but:

* ❗ **Computing the Hessian** is expensive in high dimensions
* ❗ **Inverting the Hessian** is even worse: $\mathcal{O}(n^3)$ time
* ❗ Not always guaranteed to be positive definite (can fail)

### ✅ Quasi-Newton to the Rescue

**Quasi-Newton methods** approximate the **Hessian (or its inverse)** using only **gradients**, avoiding full second derivative calculations.

---

## 🧾 Key Idea:

Instead of computing the true Hessian $H$, we build an **approximation** $B_k \approx H_k$, or more commonly its inverse $H_k \approx H_k^{-1}$, using just **first-order information (gradients)**.

---

## 📘 BFGS: Broyden–Fletcher–Goldfarb–Shanno

One of the most popular Quasi-Newton methods.

### 🧮 BFGS Update Rule

We maintain an approximation to the **inverse Hessian** $H_k$ and update it iteratively using:

Let:

* $s_k = x_{k+1} - x_k$
* $y_k = \nabla f_{k+1} - \nabla f_k$

Then:

$$
\rho_k = \frac{1}{y_k^T s_k}
$$

$$
H_{k+1} = \left( I - \rho_k s_k y_k^T \right) H_k \left( I - \rho_k y_k s_k^T \right) + \rho_k s_k s_k^T
$$

This formula updates the inverse Hessian approximation based on the curvature between two steps.

---

### 🪜 BFGS Algorithm Steps

1. Initialize $x_0$, $H_0 = I$ (identity matrix)
2. Repeat until convergence:

   * Compute gradient $\nabla f(x_k)$
   * Compute direction $p_k = -H_k \nabla f(x_k)$
   * Line search: choose step size $\alpha_k$
   * Update point: $x_{k+1} = x_k + \alpha_k p_k$
   * Update $s_k = x_{k+1} - x_k$, $y_k = \nabla f_{k+1} - \nabla f_k$
   * Update inverse Hessian approximation $H_{k+1}$ using the formula above

---

## 🪶 L-BFGS: Limited-Memory BFGS

When dimensions are huge (like in deep learning), storing $H_k \in \mathbb{R}^{n \times n}$ is infeasible.

**L-BFGS** solves this by **not storing the full matrix**, but instead keeping a limited number $m$ of $(s_k, y_k)$ pairs.

* Uses **less memory**
* Still achieves **superior convergence** compared to gradient descent

---

## 💡 Intuition

Think of BFGS like:

> “I want to take advantage of the curvature like Newton’s method, but I’ll build it myself using the gradients I’ve already computed, so I don’t have to spend time and space computing the real thing.”

And L-BFGS:

> “I’ll only remember the last few updates to save space and still approximate curvature well enough.”

---

## 🔍 Example: Minimize $f(x) = x^2 + 2x + 1$

### Step 1: Gradient and Update

* $f(x) = x^2 + 2x + 1$
* $\nabla f(x) = 2x + 2$
* Start at $x_0 = 0$
* Use BFGS update formulas (1D case simplifies)

You don't even need second derivatives. The optimizer learns the curvature from gradient changes.

---

## 🧑‍💻 Python Example Using `scipy.optimize`

```python
import numpy as np
from scipy.optimize import minimize

# Define the function
def f(x):
    return x[0]**2 + 2*x[0] + 1  # Minimum at x = -1

# Gradient of the function
def grad_f(x):
    return np.array([2*x[0] + 2])

# Run BFGS
result = minimize(f, x0=[0.0], jac=grad_f, method='BFGS')

print("BFGS Result:")
print("Minimum at x =", result.x)
```

✅ Output:

```
Minimum at x = [-1.]
```

You can change `method='BFGS'` to `method='L-BFGS-B'` to use L-BFGS.

---

## 🧪 BFGS vs L-BFGS

| Feature        | BFGS                  | L-BFGS                      |
| -------------- | --------------------- | --------------------------- |
| Memory usage   | $\mathcal{O}(n^2)$    | $\mathcal{O}(nm)$ (small)   |
| Best for       | Small/medium problems | Large-scale problems        |
| Curvature info | Full matrix           | Last m updates only         |
| Accuracy       | High                  | Slightly less, but scalable |

---

## ✅ Summary

| Term         | Meaning                                                               |
| ------------ | --------------------------------------------------------------------- |
| Quasi-Newton | Uses gradients to **approximate** the Hessian matrix                  |
| BFGS         | A specific update rule for inverse Hessian approximation              |
| L-BFGS       | A **memory-efficient** version of BFGS that stores only a few updates |
| Use case     | BFGS (small-medium problems), L-BFGS (large-scale optimization)       |

---

