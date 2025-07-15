
---

## 🧠 What is Hessian-Free Optimization?

**Hessian-Free Optimization (HF)** is a **second-order optimization** method that uses **curvature information** (like Newton’s method) but **avoids computing or storing the Hessian matrix explicitly**.

Instead, it uses **matrix-vector products with the Hessian** to estimate curvature, which makes it highly scalable — even for models with **millions of parameters**.

---

## 🚩 Why Do We Need It?

### Problems with Newton's Method:

* Newton’s method uses:

  $$
  \mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1} \nabla f(\mathbf{x}_k)
  $$

  Where $H$ is the Hessian.

* But in deep learning:

  * $H \in \mathbb{R}^{n \times n}$, with $n \sim 10^6$ or more
  * Computing and inverting this is **infeasible**

### Hessian-Free Optimization solves this by:

* 🚫 **Not computing the full Hessian**
* ✅ Using **matrix-vector products** $H \cdot v$
* ✅ Solving $H \cdot p = -\nabla f$ using **iterative methods**, like **Conjugate Gradient (CG)**

---

## 🧮 Core Idea

Let’s suppose we want to minimize a function $f(\mathbf{x})$. Newton’s method update is:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta \mathbf{x}
$$

Where:

$$
H \Delta \mathbf{x} = -\nabla f(\mathbf{x}_k)
$$

Since we don't want to compute or invert $H$, we **solve this linear system** using an **iterative method** like Conjugate Gradient (CG).

The key trick is:

* CG only needs **matrix-vector products** $H \cdot v$, **not the full matrix**

### But how do we compute $H \cdot v$ without the full Hessian?

That’s where **Pearlmutter's trick** comes in.

---

## 🧠 Pearlmutter’s Trick (Efficient Hessian-Vector Products)

For a function $f(\mathbf{x})$, the Hessian-vector product $H \cdot v$ can be computed **without forming $H$**.

Let $J$ be the Jacobian and $g = \nabla f$, then:

$$
Hv = \nabla \left( \nabla f(\mathbf{x})^T v \right)
$$

This uses **automatic differentiation (autodiff)** and is already built into frameworks like PyTorch, TensorFlow, and JAX.

---

## 🧪 Algorithm Overview (Hessian-Free)

1. **Start** with initial guess $\mathbf{x}_0$
2. **Compute** gradient $g = \nabla f(\mathbf{x}_k)$
3. **Solve** $H \cdot \Delta x = -g$ using **Conjugate Gradient (CG)**:

   * Need to compute $H \cdot v$ using Pearlmutter’s trick
4. **Update**:

   $$
   \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha \cdot \Delta x
   $$

   (possibly with a line search or trust region)

---

## ✅ Advantages

| Feature                      | Benefit                                                  |
| ---------------------------- | -------------------------------------------------------- |
| No explicit Hessian          | Saves memory and compute                                 |
| Uses curvature               | Faster convergence than 1st-order methods                |
| Works on large models        | Scales to millions of parameters (e.g., neural networks) |
| Solves quadratic subproblems | Solves them efficiently using CG                         |

---

## ⚠️ Limitations

* More complex to implement than SGD or Adam
* Slower per iteration than simple gradient methods
* Requires good preconditioners for CG to converge fast
* Still sensitive to noise in stochastic gradients (often used with batch or mini-batch)

---

## 🧑‍💻 Pseudo Code for Hessian-Free Optimization

```python
def hessian_vector_product(f, x, v):
    """
    Computes H * v without explicitly forming H
    """
    import torch
    x = x.detach().requires_grad_(True)
    g = torch.autograd.grad(f(x), x, create_graph=True)[0]
    Hv = torch.autograd.grad(g @ v, x)[0]
    return Hv

def conjugate_gradient(Hv_func, g, max_iter=50, tol=1e-6):
    """
    Solves H * p = -g using CG
    """
    x = torch.zeros_like(g)
    r = -g
    p = r.clone()
    for _ in range(max_iter):
        Hp = Hv_func(p)
        alpha = r.dot(r) / (p.dot(Hp) + 1e-10)
        x = x + alpha * p
        r_new = r - alpha * Hp
        if r_new.norm() < tol:
            break
        beta = r_new.dot(r_new) / (r.dot(r) + 1e-10)
        p = r_new + beta * p
        r = r_new
    return x
```

---

## 📘 Example Use Case: Deep Neural Networks

Hessian-Free methods were famously used in **deep learning** before Adam became dominant, especially for:

* Deep autoencoders (e.g., work by Geoffrey Hinton)
* Pretraining deep networks when SGD struggled
* Training large RNNs (e.g., Martens & Sutskever 2011)

---

## 🏁 Summary

| Term                      | Explanation                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------- |
| Hessian-Free Optimization | Uses **Hessian-vector products** to do second-order optimization without forming the Hessian |
| Pearlmutter’s Trick       | Computes $H \cdot v$ efficiently using autodiff                                              |
| Solver                    | Typically **Conjugate Gradient** for the inner loop                                          |
| Use case                  | Large models (deep nets, RNNs), where 2nd-order info helps but Hessian is too big            |

---


