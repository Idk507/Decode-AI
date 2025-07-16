
* 🧠 **Convex model** → Ridge Regression (L2-regularized Linear Regression)
* 🤖 **Non-Convex model** → A simple Neural Network

We’ll:

1. 📘 Define the problem
2. 🛠️ Implement both models
3. 📊 Compare optimization behavior
4. ⚖️ Summarize differences in performance & convergence

---

## 🧪 1. Define the Dataset: A Simple Regression Task

We'll generate a **nonlinear** dataset to highlight the differences.

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic nonlinear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X) + 0.1 * np.random.randn(*X.shape)

plt.scatter(X, y, label="Data")
plt.title("Regression Task: Nonlinear Function")
plt.show()
```

---

## ✅ 2. Convex Model: Ridge Regression

Ridge regression minimizes the **convex** cost:

$$
J(\theta) = \|y - X\theta\|^2 + \lambda \|\theta\|^2
$$

```python
from sklearn.linear_model import Ridge

# Polynomial features to increase capacity (still convex)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

ridge_model = make_pipeline(PolynomialFeatures(3), Ridge(alpha=1.0))
ridge_model.fit(X, y)
y_pred_ridge = ridge_model.predict(X)

plt.scatter(X, y, label="Data")
plt.plot(X, y_pred_ridge, color="green", label="Ridge Regression (Convex)")
plt.legend()
plt.title("Convex Optimization: Ridge Regression")
plt.show()
```

✅ Ridge regression is **fast**, **stable**, and always **converges to a global minimum**, but **can't model very complex functions** well.

---

## ❌ 3. Non-Convex Model: Neural Network

We'll use a **small neural network** (1 hidden layer) — a **non-convex** model trained via **SGD or Adam**.

```python
from sklearn.neural_network import MLPRegressor

nn_model = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh',
                        solver='adam', learning_rate_init=0.01,
                        max_iter=1000, random_state=42)

nn_model.fit(X, y.ravel())
y_pred_nn = nn_model.predict(X)

plt.scatter(X, y, label="Data")
plt.plot(X, y_pred_nn, color="red", label="Neural Network (Non-Convex)")
plt.legend()
plt.title("Non-Convex Optimization: Neural Network")
plt.show()
```

✅ Neural networks can **fit complex patterns**
❌ But optimization is **harder** — may **get stuck**, need **tuning** (e.g., learning rate, architecture)

---

## 📊 4. Compare Optimization Behaviors

| Feature                   | Ridge Regression (Convex) | Neural Network (Non-Convex)  |
| ------------------------- | ------------------------- | ---------------------------- |
| Optimization landscape    | One global minimum        | Many local minima            |
| Guarantees on convergence | Yes (global minimum)      | No (can get stuck)           |
| Model capacity            | Limited (polynomials)     | Very high (universal approx) |
| Speed of training         | Very fast                 | Slower (requires tuning)     |
| Expressive power          | Limited                   | Much more flexible           |

---

## 🧠 Insights

* **Convex optimization** is ideal when:

  * You want reliable, fast, and interpretable models
  * You know the data is relatively linear or polynomial

* **Non-convex optimization** is powerful when:

  * The data is **complex**, **nonlinear**
  * You need **representation learning** (e.g., in deep learning)
  * You're okay with spending time tuning and training

---


