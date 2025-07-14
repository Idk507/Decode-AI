
---

## 🧠 1. **What is Stochastic Gradient Descent (SGD)?**

**SGD** is an optimization algorithm used to **minimize a loss function** in machine learning and deep learning.

It is a **variant** of **Gradient Descent**, but instead of using the **entire dataset**, it uses **a single sample (or a small batch)** to update model parameters — making it **faster** and **more scalable**, especially for **large datasets**.

---

## 📉 2. **Why Gradient Descent?**

Machine learning models learn by **minimizing a loss function**.

$$
\text{Goal:} \quad \min_{\theta} J(\theta)
$$

Where:

* $\theta$: model parameters (like weights in linear regression)
* $J(\theta)$: cost/loss function (how far prediction is from actual)

We update $\theta$ using:

$$
\theta := \theta - \eta \cdot \nabla J(\theta)
$$

Where:

* $\nabla J(\theta)$: gradient of the loss function w\.r.t. $\theta$
* $\eta$: learning rate

---

## 🆚 3. **Gradient Descent vs Stochastic Gradient Descent**

| Type              | Gradient Computation | Speed     | Accuracy |
| ----------------- | -------------------- | --------- | -------- |
| Batch GD          | All data             | Slow      | Precise  |
| **Stochastic GD** | One sample           | Very fast | Noisy    |
| Mini-batch GD     | Small batch          | Balanced  | Balanced |

---

## 🔢 4. **Mathematical Formulation of SGD**

Suppose we have a dataset with $n$ training samples:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^n
$$

The overall loss (empirical risk):

$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n} L(f(x^{(i)}; \theta), y^{(i)})
$$

In SGD, for each iteration, we randomly pick **one sample** $(x^{(i)}, y^{(i)})$, and update:

$$
\theta := \theta - \eta \cdot \nabla_\theta L(f(x^{(i)}; \theta), y^{(i)})
$$

This is a **noisy but fast** estimate of the true gradient.

---

## 📘 5. **Example: SGD on Linear Regression**

Let's say we want to fit a line:

$$
\hat{y} = wx + b
$$

**Loss Function**: Mean Squared Error (MSE)

$$
L = \frac{1}{2}(y - \hat{y})^2 = \frac{1}{2}(y - (wx + b))^2
$$

**Gradients:**

$$
\frac{\partial L}{\partial w} = -(y - \hat{y}) \cdot x = -(y - (wx + b)) \cdot x
$$

$$
\frac{\partial L}{\partial b} = -(y - \hat{y}) = -(y - (wx + b))
$$

### 🔁 Update Rules:

$$
w := w - \eta \cdot \frac{\partial L}{\partial w}
$$

$$
b := b - \eta \cdot \frac{\partial L}{\partial b}
$$

---

## 💻 6. **Python Code Implementation from Scratch**

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
true_w, true_b = 3.5, 1.2
y = true_w * X + true_b + np.random.randn(100, 1) * 0.5

# Initialize weights
w = np.random.randn()
b = np.random.randn()

# Hyperparameters
learning_rate = 0.01
n_epochs = 20

# Training loop
for epoch in range(n_epochs):
    for i in range(len(X)):
        xi = X[i]
        yi = y[i]
        
        # Prediction
        y_pred = w * xi + b
        
        # Compute gradients
        error = y_pred - yi
        dw = error * xi
        db = error
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
    
    print(f"Epoch {epoch+1}: w = {w:.4f}, b = {b:.4f}")

# Plot
plt.scatter(X, y, label="Data")
plt.plot(X, w*X + b, color='r', label="Fitted Line")
plt.legend()
plt.title("Linear Regression using SGD")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

---

## 📊 7. **Comparison to Batch Gradient Descent**

```python
# Batch GD update
for epoch in range(n_epochs):
    y_pred = w * X + b
    error = y_pred - y
    dw = (2/len(X)) * np.sum(error * X)
    db = (2/len(X)) * np.sum(error)
    
    w -= learning_rate * dw
    b -= learning_rate * db
```

Notice how **batch GD** uses full data for each epoch, while **SGD** updates after each sample.

---

## ✅ 8. **Advantages of SGD**

* Faster for large datasets
* Online learning possible
* Good for non-convex optimization (deep learning)

---

## ⚠️ 9. **Disadvantages**

* Noisy updates → may oscillate
* May not converge to exact minimum
* Requires good tuning of learning rate

---

## 🧪 10. **SGD in PyTorch / scikit-learn**

### 🔹 PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 📌 Summary

| Concept         | Explanation                                  |
| --------------- | -------------------------------------------- |
| **SGD**         | Optimizes model using gradient of one sample |
| **Update Rule** | $\theta := \theta - \eta \cdot \nabla L$     |
| **Use case**    | Training on large datasets, deep learning    |
| **Pros**        | Fast, scalable, supports online learning     |
| **Cons**        | Noisy, sensitive to learning rate            |

---


