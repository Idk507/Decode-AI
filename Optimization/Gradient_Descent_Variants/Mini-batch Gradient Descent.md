

---

## ✅ 1. What is Mini-batch Gradient Descent?

Mini-batch Gradient Descent is a **hybrid approach** between:

| Type                                  | Description                                                          |
| ------------------------------------- | -------------------------------------------------------------------- |
| **Batch Gradient Descent**            | Uses **all data** to compute the gradient — accurate but slow        |
| **Stochastic Gradient Descent (SGD)** | Uses **only one sample** per update — fast but noisy                 |
| **Mini-batch Gradient Descent**       | Uses a **small batch** (e.g., 32 or 64) per update — balance of both |

It updates the model **more frequently than batch GD**, but **less noisy than SGD**.

---

## 🔢 2. Mathematical Formulation

Suppose you have:

* Dataset with $n$ samples: $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$
* A model with parameters $\theta$
* A loss function $L(f(x; \theta), y)$
* Mini-batch size: $m \ll n$

### Mini-batch Gradient Update:

At each iteration, randomly sample a **mini-batch** $B = \{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$, then:

$$
\nabla J_B(\theta) = \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta L(f(x^{(i)}; \theta), y^{(i)})
$$

Then update parameters:

$$
\theta := \theta - \eta \cdot \nabla J_B(\theta)
$$

Where:

* $\eta$: learning rate
* $m$: mini-batch size (commonly 16, 32, 64, 128)

---

## 💡 3. Why Use Mini-batch Gradient Descent?

| Feature         | Mini-Batch GD                                     |
| --------------- | ------------------------------------------------- |
| Speed           | ✅ Faster than full batch                          |
| Noise           | ✅ Smoother than SGD                               |
| Memory usage    | ✅ Balanced (uses only part of the data at a time) |
| Parallelization | ✅ Supports GPUs & vectorization well              |

---

## 🧠 4. Intuitive Example

Let’s say we are training a model on **1000 samples**.

* **Batch GD**: Uses all 1000 samples → 1 update per epoch
* **SGD**: Uses 1 sample at a time → 1000 updates per epoch
* **Mini-batch GD** (batch size = 100): 10 updates per epoch

This gives a sweet spot: **enough updates per epoch** but without too much variance.

---

## 💻 5. Mini-batch Gradient Descent from Scratch (Linear Regression)

### Step-by-step Implementation:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
true_w, true_b = 3.5, 1.2
y = true_w * X + true_b + np.random.randn(1000, 1) * 0.5

# Initialize parameters
w = np.random.randn()
b = np.random.randn()

# Hyperparameters
learning_rate = 0.01
n_epochs = 20
batch_size = 64

# Training loop
for epoch in range(n_epochs):
    indices = np.random.permutation(len(X))  # Shuffle data
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Prediction
        y_pred = w * X_batch + b
        error = y_pred - y_batch
        
        # Gradients
        dw = (1 / len(X_batch)) * np.sum(error * X_batch)
        db = (1 / len(X_batch)) * np.sum(error)
        
        # Parameter update
        w -= learning_rate * dw
        b -= learning_rate * db
    
    print(f"Epoch {epoch+1}: w = {w:.4f}, b = {b:.4f}")

# Plot final model
plt.scatter(X, y, alpha=0.3)
plt.plot(X, w*X + b, color='red', label='Fitted Line')
plt.title('Mini-batch GD on Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

---

## ⚙️ 6. Using Mini-batch GD in PyTorch

PyTorch automatically handles mini-batch using `DataLoader`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Prepare data
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define model
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(20):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 📊 7. Summary: Batch vs Mini-batch vs SGD

| Feature          | Batch GD | Mini-batch GD | SGD             |
| ---------------- | -------- | ------------- | --------------- |
| Speed per update | ❌ Slow   | ✅ Medium      | ✅ Fast          |
| Noise            | ✅ Stable | ✅ Balanced    | ❌ Noisy         |
| Memory usage     | ❌ High   | ✅ Medium      | ✅ Low           |
| Convergence      | ✅ Stable | ✅ Good        | ❌ May Oscillate |

---

## 📌 Real-world Use

* Almost **all deep learning** frameworks (PyTorch, TensorFlow) use **mini-batch gradient descent**
* Popular batch sizes: **32, 64, 128**
* Mini-batch works great with **GPUs**, allowing matrix operations and parallelism

---


