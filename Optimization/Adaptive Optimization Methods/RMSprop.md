RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm that solves a key problem with basic gradient descent: different parameters often need different learning rates to converge efficiently.

## The Problem RMSprop Solves

In standard gradient descent, we use the same learning rate for all parameters. But imagine optimizing a function that's steep in one direction and shallow in another - like a narrow valley. A large learning rate might cause oscillations in the steep direction, while a small learning rate makes progress painfully slow in the shallow direction.


## 📌 What Is RMSProp?

**RMSProp** is an **adaptive learning rate optimization algorithm** designed to fix a major problem in AdaGrad — the **shrinking learning rate** over time.

It adjusts the learning rate **per parameter** like AdaGrad, but instead of accumulating **all past squared gradients**, it uses a **moving average** of recent squared gradients.

---

## ❗ Motivation: Why RMSProp?

### 📉 Problem with AdaGrad:

In AdaGrad:

$$
G_t = G_{t-1} + g_t^2
$$

This sum keeps growing → the denominator in AdaGrad’s update rule:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t
$$

→ becomes **too large**, and the effective learning rate **approaches zero**
⛔ Model stops learning

---

## 💡 RMSProp’s Fix:

Instead of summing all squared gradients, **RMSProp uses exponential moving average (EMA)** of squared gradients.

---

## 🧮 1. RMSProp: Mathematical Formulation

Let:

* $g_t$: gradient of parameter at time $t$
* $E[g^2]_t$: moving average of squared gradients
* $\gamma$: decay rate (e.g., 0.9)
* $\eta$: learning rate
* $\epsilon$: small constant for numerical stability

### 🔁 Update Rule:

#### 1. Compute squared gradient moving average:

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma)g_t^2
$$

#### 2. Parameter update:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

---

### 📌 Important Points:

* $E[g^2]_t$: tracks recent squared gradients (not all past ones like AdaGrad)
* Keeps learning rate **adaptive and stable**
* Avoids AdaGrad’s issue of vanishing learning rate

---

## 🧠 2. Intuition Behind RMSProp

| Feature          | AdaGrad                     | RMSProp                            |
| ---------------- | --------------------------- | ---------------------------------- |
| Gradient history | All past gradients          | Recent gradients (EMA)             |
| Learning rate    | Shrinks forever             | Stabilized over time               |
| Suitability      | Sparse features (NLP, etc.) | Deep networks, non-convex problems |

---

## 🔢 3. Step-by-Step Algorithm

Let’s go through step by step for **one parameter $\theta$**.

### Initialization:

* $\theta_0 = \text{initial value}$
* $E[g^2]_0 = 0$
* Learning rate $\eta = 0.01$
* Decay rate $\gamma = 0.9$
* $\epsilon = 1e-8$

---

### At each step $t$:

1. Compute gradient $g_t$
   → e.g., $g_t = \frac{\partial L}{\partial \theta}$

2. Update EMA of squared gradients:

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma)g_t^2
$$

3. Compute adjusted learning rate:

$$
\text{Effective LR} = \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}
$$

4. Update parameter:

$$
\theta_{t+1} = \theta_t - \text{Effective LR} \cdot g_t
$$

---

## 🔍 4. Manual Example: Minimize $f(x) = x^2$

Let’s take:

* Initial $x = 4.0$
* Learning rate $\eta = 0.1$
* Decay $\gamma = 0.9$
* $\epsilon = 1e-8$
* $f(x) = x^2 \Rightarrow f'(x) = 2x$

### Step 1:

* $x = 4$
* $g = 8$
* $E[g^2]_1 = 0.9 \cdot 0 + 0.1 \cdot 64 = 6.4$
* Update:

$$
x = 4 - \frac{0.1}{\sqrt{6.4}} \cdot 8 \approx 4 - 0.1/2.53 \cdot 8 \approx 4 - 0.316 \cdot 8 = 1.47
$$

(approximate)

### Step 2:

* $g = 2x = 2.94$
* $g^2 = 8.64$
* $E[g^2]_2 = 0.9 \cdot 6.4 + 0.1 \cdot 8.64 = 6.624$
* New learning rate: $0.1 / \sqrt{6.624} \approx 0.1 / 2.57 = 0.0389$
* Update x again

And so on...

---

## 💻 5. Python Implementation (From Scratch)

```python
import numpy as np

# Function and gradient
def f(x):
    return x**2

def grad(x):
    return 2 * x

# Hyperparameters
x = 4.0
eta = 0.1
gamma = 0.9
epsilon = 1e-8
Eg2 = 0  # E[g^2]_t
n_steps = 30

print("Step | x       | Gradient | E[g^2]   | Effective LR | Update")
for t in range(1, n_steps+1):
    g = grad(x)
    Eg2 = gamma * Eg2 + (1 - gamma) * g**2
    lr = eta / (np.sqrt(Eg2) + epsilon)
    update = lr * g
    x -= update
    print(f"{t:>4} | {x:7.4f} | {g:8.4f} | {Eg2:8.4f} | {lr:12.6f} | {update:7.4f}")
```

---

## 📊 6. RMSProp in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Toy data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Model
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)

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

## 📚 7. Summary: RMSProp Key Points

| Property   | RMSProp                             |
| ---------- | ----------------------------------- |
| Type       | Adaptive learning rate optimizer    |
| Fixes      | AdaGrad's vanishing learning rate   |
| Method     | Moving average of squared gradients |
| Parameters | $\eta$, $\gamma$, $\epsilon$        |
| Strength   | Deep learning, non-convex functions |
| Limitation | No momentum (fixed in Adam)         |

---

## 🧠 TL;DR

* RMSProp scales the learning rate **individually per parameter**.
* It uses an **exponential moving average** of squared gradients.
* It **stabilizes learning** and works well in practice.
* It’s **commonly used in training RNNs, CNNs**, and deep learning models.


## How RMSprop Works

RMSprop adapts the learning rate for each parameter individually by keeping track of the magnitude of recent gradients.

### Step 1: Initialize Variables
For each parameter, we maintain:
- The parameter itself: `θ`
- A moving average of squared gradients: `v` (initialized to 0)
- A small constant `ε` (typically 1e-8) to prevent division by zero
- A decay rate `β` (typically 0.9)
- A learning rate `α` (typically 0.001)

### Step 2: Compute Gradients
Calculate the gradient of the loss function with respect to each parameter:
```
g = ∇L(θ)
```

### Step 3: Update Moving Average of Squared Gradients
This is the key innovation - we track how "active" each parameter has been recently:
```
v = β × v + (1 - β) × g²
```

Think of `v` as a "memory" of recent gradient magnitudes. The decay rate `β` controls how much we remember from the past versus the current gradient.

### Step 4: Compute Adaptive Learning Rate
For each parameter, we scale the learning rate by the square root of this moving average:
```
θ = θ - α × g / (√v + ε)
```

The `√v` term acts as a normalizer - parameters with large gradients get smaller effective learning rates, while parameters with small gradients get relatively larger effective learning rates.

## Why This Works

**Automatic scaling**: If a parameter has been receiving large gradients (meaning `v` is large), then `√v` is large, so the effective learning rate `α/(√v + ε)` becomes smaller. This prevents overshooting.

**Consistent progress**: If a parameter has been receiving small gradients (meaning `v` is small), then `√v` is small, so the effective learning rate becomes larger. This accelerates progress in slowly-changing dimensions.

**Memory of past gradients**: The exponential moving average means recent gradients matter more than old ones, but we don't completely forget the past. This provides stability while adapting to changing conditions.

## Example in Action

Imagine training a neural network where:
- Weight A has been receiving gradients of magnitude 0.1
- Weight B has been receiving gradients of magnitude 10.0

After several iterations:
- For Weight A: `v_A ≈ 0.01`, so effective learning rate ≈ `α/0.1`
- For Weight B: `v_B ≈ 100`, so effective learning rate ≈ `α/10`

Weight A gets 100× higher effective learning rate than Weight B, helping both converge at similar rates despite very different gradient magnitudes.

## Advantages of RMSprop

- **Automatic learning rate adaptation**: No need to manually tune learning rates for different parameters
- **Handles sparse gradients well**: Common in natural language processing
- **Reduces oscillations**: Especially helpful in narrow valleys of the loss landscape
- **Simple to implement**: Just a few extra lines compared to basic gradient descent

## Limitations

- **Still requires tuning**: The base learning rate `α` and decay rate `β` need to be set
- **Can be too aggressive**: Sometimes reduces learning rates too much, slowing convergence
- **No momentum**: Unlike some other optimizers, RMSprop doesn't maintain momentum from previous updates

RMSprop was a significant step forward in optimization and laid the groundwork for even more sophisticated optimizers like Adam, which combines RMSprop's adaptive learning rates with momentum.
