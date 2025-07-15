
---

## 📌 What is AdaGrad?

**AdaGrad (Adaptive Gradient Algorithm)** is an optimization algorithm that **adapts the learning rate** for each parameter **individually** during training.

It is particularly useful when:

* Features are sparse (like in NLP, one-hot vectors)
* Some parameters require larger/smaller updates than others

---

## 🧠 1. Intuition Behind AdaGrad

In standard **Gradient Descent**, we use:

$$
\theta := \theta - \eta \cdot \nabla_\theta J(\theta)
$$

* Same learning rate $\eta$ for **all parameters**
* Doesn’t adapt based on parameter importance or gradient history

### 🔍 Problem:

* If gradients vary a lot, a **fixed learning rate** may:

  * Overshoot minima (for large gradients)
  * Converge slowly (for small gradients)

---

## 💡 Solution: AdaGrad

AdaGrad **adapts the learning rate** using the **accumulated squared gradients**.

The idea is:

* Parameters that have **frequently had large gradients** → get **smaller learning rates**
* Parameters with **small gradients** → get **larger learning rates**

This helps:
✅ Stabilize learning
✅ Improve convergence on sparse features

---

## 🔢 2. Mathematical Formulation of AdaGrad

Let’s define:

* $g_t$: gradient at time step $t$
* $G_t$: sum of squared gradients until time $t$
* $\theta_t$: model parameter at time $t$
* $\eta$: initial learning rate (scalar)
* $\epsilon$: small constant for numerical stability

### 🧮 AdaGrad update rule (per parameter):

$$
G_t = G_{t-1} + g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t
$$

So, the **effective learning rate** becomes:

$$
\frac{\eta}{\sqrt{G_t + \epsilon}}
$$

👉 Over time, if a parameter sees **large gradients repeatedly**, its learning rate **decays**.

---

## 🧪 3. Toy Example (2D Optimization)

Imagine a loss function:

$$
f(w, b) = w^2 + 10b^2
$$

* The slope w\.r.t $b$ is steeper → gradients in $b$ direction are large
* AdaGrad will **reduce learning rate** for $b$ more quickly than for $w$
* This **balances the steps** toward minimum

---

## 💻 4. AdaGrad Python Implementation (From Scratch)

### Objective: Minimize $f(x) = x^2$

```python
import numpy as np
import matplotlib.pyplot as plt

# Function and gradient
def f(x):
    return x**2

def grad(x):
    return 2 * x

# AdaGrad parameters
x = 5.0
learning_rate = 0.1
epsilon = 1e-8
G = 0
n_steps = 50

x_history = [x]

for t in range(1, n_steps+1):
    g = grad(x)
    G += g**2
    adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
    x -= adjusted_lr * g
    x_history.append(x)

# Plot convergence
plt.plot(x_history, label="x values over steps")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.title("AdaGrad Convergence on f(x) = x²")
plt.grid(True)
plt.legend()
plt.show()

print(f"Final value of x: {x:.4f}")
```

🟢 Observe how:

* Learning rate shrinks over time
* Converges rapidly at first, then slows down

---

## 🔧 5. Pros and Cons of AdaGrad

### ✅ Pros

* Good for **sparse data** (e.g., NLP, recommender systems)
* No need to manually tune learning rate per parameter
* Fast initial convergence

### ❌ Cons

* **Learning rate shrinks too much over time**

  * This may cause the model to **stop learning prematurely**
* Doesn’t “forget” old gradients

---

## 🔁 6. AdaGrad vs Other Optimizers

| Optimizer   | Adapts LR? | Momentum?               | Description                                    |
| ----------- | ---------- | ----------------------- | ---------------------------------------------- |
| SGD         | ❌ No       | ❌ No                    | Fixed learning rate                            |
| Momentum    | ❌ No       | ✅ Yes                   | Accelerates using past updates                 |
| **AdaGrad** | ✅ Yes      | ❌ No                    | Scales learning rate by past squared gradients |
| **RMSProp** | ✅ Yes      | ❌ (uses moving average) | Solves AdaGrad's decay issue                   |
| **Adam**    | ✅ Yes      | ✅ Yes                   | Combines RMSProp + Momentum                    |

---

## ⚙️ 7. AdaGrad in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Toy dataset
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Model
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.1)

# Training
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 🎯 Real World Use Case

* **Text classification**, **language modeling**, **recommender systems** with high-dimensional sparse features
* Example: Word2Vec used AdaGrad in its early implementations

---
---

## 🧠 1. Core Idea Recap (Intuition)

In vanilla SGD:

$$
\theta := \theta - \eta \cdot \nabla_\theta J(\theta)
$$

> The **same learning rate** $\eta$ is used for every parameter and every step.

### ✅ Problem:

* Features or parameters that appear frequently → receive many updates
* Others → fewer updates
* So we **need to scale learning rate** per parameter.

### ✅ AdaGrad's Fix:

* Accumulate the **squared gradients**
* Scale learning rate by dividing by the square root of this sum

---

## 🧮 2. Mathematical Foundation

Let:

* $g_{t,i}$: gradient of parameter $\theta_i$ at time $t$
* $G_{t,i}$: sum of squared gradients for $\theta_i$ until time $t$

### Gradient Accumulation:

$$
G_{t,i} = \sum_{\tau=1}^{t} g_{\tau, i}^2
$$

### Parameter Update Rule:

$$
\theta_{t+1, i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,i} + \epsilon}} \cdot g_{t,i}
$$

Where:

* $\eta$: initial learning rate
* $\epsilon$: small value to avoid division by zero (e.g., $10^{-8}$)

---

## 🔁 3. Step-by-Step Algorithm (with Math)

For **each parameter $\theta_i$**:

### Step 1: Initialize

* $\theta_i = \text{some value}$
* $G_i = 0$ (accumulated squared gradient)

---

### Step 2: For each time step $t$

1. Compute gradient:

   $$
   g_{t,i} = \frac{\partial L}{\partial \theta_i}
   $$

2. Accumulate squared gradient:

   $$
   G_{i} := G_i + g_{t,i}^2
   $$

3. Adjusted learning rate:

   $$
   \hat{\eta}_i := \frac{\eta}{\sqrt{G_i + \epsilon}}
   $$

4. Update parameter:

   $$
   \theta_i := \theta_i - \hat{\eta}_i \cdot g_{t,i}
   $$

Repeat until convergence.

---

## 🔢 4. Numerical Example (Manual Calculation)

### Objective: Minimize $f(x) = x^2$

#### Initialization:

* $x = 4.0$
* $\eta = 1.0$
* $G = 0$
* $\epsilon = 1e-8$
* Gradient: $f'(x) = 2x$

---

### 🔄 Step-by-step for 3 iterations

#### **Step 1:**

* $x = 4.0$
* $g = 2 \cdot 4 = 8$
* $G = 0 + 8^2 = 64$
* $\hat{\eta} = 1 / \sqrt{64 + 1e-8} = 1 / 8 = 0.125$
* Update:

  $$
  x = 4 - 0.125 \cdot 8 = 4 - 1 = 3
  $$

---

#### **Step 2:**

* $x = 3.0$
* $g = 2 \cdot 3 = 6$
* $G = 64 + 36 = 100$
* $\hat{\eta} = 1 / \sqrt{100 + 1e-8} = 1 / 10 = 0.1$
* Update:

  $$
  x = 3 - 0.1 \cdot 6 = 3 - 0.6 = 2.4
  $$

---

#### **Step 3:**

* $x = 2.4$
* $g = 2 \cdot 2.4 = 4.8$
* $G = 100 + 23.04 = 123.04$
* $\hat{\eta} = 1 / \sqrt{123.04} \approx 0.090$
* Update:

  $$
  x = 2.4 - 0.090 \cdot 4.8 \approx 2.4 - 0.432 = 1.968
  $$

🎯 You can see: as **gradient squared accumulates**, **learning rate decreases**.

---

## 💻 5. Full Step-by-Step AdaGrad in Python

```python
import numpy as np

# Objective function: f(x) = x^2
def f(x):
    return x**2

# Gradient: f'(x) = 2x
def grad(x):
    return 2 * x

# Initialize
x = 4.0
eta = 1.0
epsilon = 1e-8
G = 0

# Run for 10 steps
print("Step | x       | Gradient | G        | Learning Rate | Update")
for t in range(1, 11):
    g = grad(x)
    G += g**2
    adjusted_eta = eta / (np.sqrt(G) + epsilon)
    update = adjusted_eta * g
    x -= update
    
    print(f"{t:>4} | {x:7.4f} | {g:8.4f} | {G:8.4f} | {adjusted_eta:13.6f} | {update:7.4f}")
```

---

## 🧠 Summary: How AdaGrad Works

| Step                   | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| 1. Init                | Set $G_i = 0$, initial $\theta_i$, learning rate $\eta$ |
| 2. Compute Gradient    | $g_{t,i} = \partial L / \partial \theta_i$              |
| 3. Accumulate          | $G_i = G_i + g_{t,i}^2$                                 |
| 4. Scale Learning Rate | $\hat{\eta}_i = \eta / \sqrt{G_i + \epsilon}$           |
| 5. Update              | $\theta_i = \theta_i - \hat{\eta}_i \cdot g_{t,i}$      |

---

## ✅ When to Use AdaGrad

* Sparse features (NLP, recommender systems)
* High-dimensional data
* Want per-parameter adaptive learning

---
## 🧠 Summary

| Feature       | Description                             |
| ------------- | --------------------------------------- |
| Name          | AdaGrad (Adaptive Gradient)             |
| Learns rate   | Decays over time per parameter          |
| Core idea     | Penalize frequently-updated parameters  |
| Works well on | Sparse, high-dimensional data           |
| Problem       | Learning rate decays too much over time |
| Fix           | Use **RMSProp** or **Adam** to solve it |


