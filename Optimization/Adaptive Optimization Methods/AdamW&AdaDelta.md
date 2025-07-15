
Great! Let's break down **AdamW**, **AdaDelta**, and **Nadam** optimizers in **simple terms**, along with the **math behind each**, step-by-step. These are all improvements over standard gradient descent, aiming to make training **faster and more stable**.

---

## 🧠 1. **AdamW (Adam with Weight Decay)**

### 🔹 What is it?

AdamW is a modification of the **Adam optimizer** that **decouples weight decay** (L2 regularization) from the gradient updates. It helps in **better generalization**.

> 🔑 Problem with Adam: Regularization (L2) is added **inside** the gradient update, which interacts badly with adaptive learning rates.

> ✅ Fix in AdamW: Applies **weight decay separately** (after the gradient update), just like SGD.

---

### 🔣 Math Behind AdamW

Let’s define:

* $\theta_t$: Parameters at step t
* $g_t = \nabla_\theta L(\theta_t)$: Gradient
* $m_t$: 1st moment (mean of gradients)
* $v_t$: 2nd moment (uncentered variance)
* $\beta_1, \beta_2$: Decay rates (e.g., 0.9, 0.999)
* $\eta$: Learning rate
* $\lambda$: Weight decay factor

#### 🔁 Step-by-step:

1. **Update biased moment estimates:**

   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$

   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$

2. **Bias correction:**

   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$

3. **Update parameters with weight decay (decoupled):**

   $$
   \theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
   $$

> 👇 Unlike regular Adam, the `+ λθ_t` part is added **after** adaptive updates.

---

## 🔁 2. **AdaDelta**

### 🔹 What is it?

AdaDelta fixes the **learning rate decay problem** in AdaGrad. It **adapts learning rates** using only **a window of past gradients**, not accumulating forever.

> 🔑 AdaGrad’s issue: Squared gradients accumulate → Learning rate goes to zero.
> ✅ AdaDelta uses an **exponential moving average** → Keeps learning rates stable.

---

### 🔣 Math Behind AdaDelta

Let’s define:

* $E[g^2]_t$: Running average of squared gradients
* $E[\Delta \theta^2]_t$: Running average of squared parameter updates
* $\rho$: Decay rate (e.g., 0.9)
* $\epsilon$: Small constant for numerical stability

#### 🔁 Step-by-step:

1. **Gradient square EMA:**

   $$
   E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2
   $$

2. **Compute update:**

   $$
   \Delta \theta_t = - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
   $$

3. **Update EMA of update step:**

   $$
   E[\Delta \theta^2]_t = \rho E[\Delta \theta^2]_{t-1} + (1 - \rho) (\Delta \theta_t)^2
   $$

4. **Apply update:**

   $$
   \theta_{t+1} = \theta_t + \Delta \theta_t
   $$

> 🎯 AdaDelta dynamically adjusts the size of the updates without requiring a global learning rate.

---

## 🧮 3. **Nadam (Nesterov + Adam)**

### 🔹 What is it?

Nadam combines **Adam** with **Nesterov Accelerated Gradient (NAG)**, which looks ahead before updating. It gives a small performance boost by **correcting the momentum step**.

> 🔑 NAG: Look ahead before calculating gradient
> ✅ Nadam: Applies this idea inside Adam’s adaptive update

---

### 🔣 Math Behind Nadam

Let’s define:

* Same symbols as Adam: $g_t, m_t, v_t, \beta_1, \beta_2$

#### 🔁 Step-by-step:

1. **Update moment estimates:**

   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$

   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$

2. **Bias correction:**

   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$

3. **Look-ahead momentum (Nesterov style):**

   $$
   \tilde{m}_t = \beta_1 \hat{m}_t + \left( \frac{(1 - \beta_1) g_t}{1 - \beta_1^t} \right)
   $$

4. **Update rule:**

   $$
   \theta_{t+1} = \theta_t - \eta \cdot \frac{\tilde{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   $$

> 🔁 Nadam adds a **correction** for momentum ahead of time.

---

## 🧾 Summary Table

| Optimizer    | Key Feature                    | Gradient Memory   | Learning Rate Adaptation | Regularization |
| ------------ | ------------------------------ | ----------------- | ------------------------ | -------------- |
| **AdamW**    | Weight decay decoupled         | Mean + Variance   | Yes                      | ✅ Decoupled L2 |
| **AdaDelta** | No manual LR, adaptive updates | Grad history only | Yes (no LR needed)       | ❌              |
| **Nadam**    | Nesterov + Adam                | Mean + Variance   | Yes                      | ❌              |

---



$$
f(\theta) = \theta^2 \quad \Rightarrow \quad \nabla f(\theta) = 2\theta
$$

---

## ✅ 1. **AdamW Implementation (from scratch)**

```python
import numpy as np

def adamw(epochs=100, eta=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
    theta = 5.0  # Initial parameter
    m, v = 0, 0

    for t in range(1, epochs + 1):
        grad = 2 * theta  # ∇f(θ) = 2θ

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # AdamW update
        theta -= eta * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * theta)

        print(f"Epoch {t:03d} | θ: {theta:.5f}")

adamw()
```

---

## ✅ 2. **AdaDelta Implementation (from scratch)**

```python
def adadelta(epochs=100, rho=0.95, epsilon=1e-6):
    theta = 5.0
    Eg2 = 0  # EMA of squared gradients
    Edx2 = 0  # EMA of squared parameter updates

    for t in range(1, epochs + 1):
        grad = 2 * theta

        Eg2 = rho * Eg2 + (1 - rho) * grad ** 2
        delta = - np.sqrt(Edx2 + epsilon) / np.sqrt(Eg2 + epsilon) * grad

        Edx2 = rho * Edx2 + (1 - rho) * delta ** 2
        theta += delta

        print(f"Epoch {t:03d} | θ: {theta:.5f}")

adadelta()
```

---

## ✅ 3. **Nadam Implementation (from scratch)**

```python
def nadam(epochs=100, eta=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    theta = 5.0
    m, v = 0, 0

    for t in range(1, epochs + 1):
        grad = 2 * theta

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Nesterov momentum correction
        m_nesterov = beta1 * m_hat + ((1 - beta1) * grad / (1 - beta1 ** t))

        theta -= eta * m_nesterov / (np.sqrt(v_hat) + epsilon)

        print(f"Epoch {t:03d} | θ: {theta:.5f}")

nadam()
```

---

### 📌 Notes:

* These examples assume optimizing a **single variable** (scalar). You can easily extend to vector parameters by replacing scalars with arrays.
* For real neural networks, these are implemented per weight/bias in each layer.
* The `print` logs show how the parameter θ converges to 0 (the minimum of $\theta^2$).

---

