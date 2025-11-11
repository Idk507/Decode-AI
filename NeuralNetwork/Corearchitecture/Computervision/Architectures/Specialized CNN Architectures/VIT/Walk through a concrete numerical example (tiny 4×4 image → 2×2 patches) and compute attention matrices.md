
---

## Notation & shapes (single head)

* Sequence length (tokens): (N).
* Input embedding dimension: (D).
* Per-head dimension: (d) (so queries/keys/values are (d)-dim).
* Query, Key, Value matrices (after linear projections):
  [
  Q,;K,;V \in \mathbb{R}^{N\times d}.
  ]
* Scaled score matrix:
  [
  S = \frac{QK^\top}{\sqrt{d}} \in \mathbb{R}^{N\times N}.
  ]
  (we use row (i) as query (i), columns index keys)
* Attention weights (row-wise softmax):
  [
  A = \operatorname{softmax}(S) \in \mathbb{R}^{N\times N},
  ]
  where each row $(A_{i,:} = \operatorname{softmax}(S_{i,:}))$ sums to 1.
* Attention output:
  [
  O = A,V \in \mathbb{R}^{N\times d}.
  ]
* A scalar loss (L) depends (via later network layers) on (O). We assume we have $(\frac{\partial L}{\partial O})$, denote it $(G_O \in \mathbb{R}^{N\times d})$.

Goal: compute
[
\frac{\partial L}{\partial A},\quad
\frac{\partial L}{\partial S},\quad
\frac{\partial L}{\partial Q},\quad
\frac{\partial L}{\partial K},\quad
\frac{\partial L}{\partial V}.
]
(And then optionally $(\partial L/\partial W_Q,\partial L/\partial W_K,\partial L/\partial W_V)$ if $(Q=XW_Q), etc.)$

---

## Step 0 — From output gradient to immediate upstream gradients

Because (O = A V), treat (A) and (V) as inputs to this matrix product.

1. Gradient w.r.t. (A):
   [
   G_A \equiv \frac{\partial L}{\partial A} = G_O,V^\top \in \mathbb{R}^{N\times N}.
   ]
   (Reason: elementwise, $(O_{i,:} = \sum_{j} A_{i,j} V_{j,:})$, so $(\partial L/\partial A_{i,j} = \sum_{k} (\partial L/\partial O_{i,k}) V_{j,k})$.)

2. Gradient w.r.t. (V):
   [
   G_V \equiv \frac{\partial L}{\partial V} = A^\top,G_O \in \mathbb{R}^{N\times d}.
   ]
   (Because (O = A V), gradient flows back to V by multiplying left by $(A^\top)$.)

So far:
[
G_A = G_O V^\top,\qquad G_V = A^\top G_O.
]

---

## Step 1 — Backprop through softmax (row-wise)

Recall each row $(a_i = A_{i,:} = \operatorname{softmax}(s_i))$ where (s_i) is row (i) of (S). For a single row (i), with upstream gradient $(g^{(a)}_i = G_A[i,:])$ (row vector length (N)), the derivative w.r.t. (s_i) is:

[
\frac{\partial L}{\partial s_i}
= J_{softmax}(s_i)^\top , g^{(a)}*i,
]
where
[
J*{softmax}(s_i) = \operatorname{diag}(a_i) - a_i a_i^\top.
]

This yields the compact and numerically stable formula (elementwise form):

For each row (i) and column (j):
[
\big(G_S\big)*{i,j}
\equiv \frac{\partial L}{\partial S*{i,j}}
= a_{i,j}\Big( g^{(a)}*{i,j} - \sum*{k=1}^N g^{(a)}*{i,k},a*{i,k} \Big).
]

Vector form for row (i):
[
G_{S}[i,:] = a_i \odot \Big( g^{(a)}_i - \big(a_i \cdot g^{(a)}*i{}^\top\big) \mathbf{1} \Big),
]
where (\odot) is elementwise multiply, and $(\mathbf{1})$ is a length-(N) vector of ones. More simply:
[
G*{S}[i,:] = \big(\operatorname{diag}(a_i) - a_i a_i^\top\big),g^{(a)}_i^\top .
]

Stack across all rows: compute $(G_S\in\mathbb{R}^{N\times N})$ rowwise by the above.

So:
[
G_S = \operatorname{softmax_jacobian_mul}(A,,G_A).
]

(Implementation tip: compute rowwise scalar $(c_i = a_i^\top g^{(a)}_i)$ then $(G_S[i,:]=a_i\odot (g^{(a)}_i - c_i))$.)

---

## Step 2 — Backprop through scaled dot-product $(S = \frac{QK^\top}{\sqrt{d}})$

We have
[
S = \frac{1}{\sqrt{d}} Q K^\top.
]
Differentiate:

* Gradient w.r.t. (Q):
  [
  G_Q \equiv \frac{\partial L}{\partial Q}
  = \frac{1}{\sqrt{d}},G_S,K \quad \in\mathbb{R}^{N\times d}.
  ]
  (Because (\partial \operatorname{tr}(G_S^\top Q K^\top)/\partial Q = G_S K).)

* Gradient w.r.t. (K):
  [
  G_K \equiv \frac{\partial L}{\partial K}
  = \frac{1}{\sqrt{d}},G_S^\top,Q \quad \in\mathbb{R}^{N\times d}.
  ]

Proof sketch: elementwise,
[
S_{i,j} = \frac{1}{\sqrt{d}} \sum_{r=1}^{d} Q_{i,r}K_{j,r}.
]
So $(\partial L/\partial Q_{i,r} = \sum_j (\partial L/\partial S_{i,j}) \cdot \frac{1}{\sqrt{d}} K_{j,r})$, which in matrix form is above.

So summarizing the chain:

1. (G_O) given.
2. (G_A = G_O V^\top).
3. (G_V = A^\top G_O).
4. (G_S = \text{softmax_backprop}(S,A,G_A)).
5. (G_Q = \frac{1}{\sqrt{d}} G_S K).
6. (G_K = \frac{1}{\sqrt{d}} G_S^\top Q).

---

## Step 3 — If (Q,K,V) are produced by linear layers from inputs

Commonly $(Q = XW_Q,; K = XW_K,; V = XW_V)$ where $(X\in\mathbb{R}^{N\times D_{in}})$ is the input sequence and $(W_Q,W_K,W_V\in\mathbb{R}^{D_{in}\times d})$.

Then gradients w.r.t. the learnable weight matrices are:

* $(\displaystyle \frac{\partial L}{\partial W_Q} = X^\top G_Q \in \mathbb{R}^{D_{in}\times d}.)$
* $(\displaystyle \frac{\partial L}{\partial W_K} = X^\top G_K \in \mathbb{R}^{D_{in}\times d}.)$
* $(\displaystyle \frac{\partial L}{\partial W_V} = X^\top G_V \in \mathbb{R}^{D_{in}\times d}.)$

And gradient to the input (X) accumulates contributions from all three projections:

[
\frac{\partial L}{\partial X} = G_Q W_Q^\top + G_K W_K^\top + G_V W_V^\top \in\mathbb{R}^{N\times D_{in}}.
]

(If the projections are separate per head or implemented as one big projection (XW) that is reshaped, the same matrix calculus applies with appropriate reshapes.)

---

## Step 4 — Multi-head attention notes

For multi-head attention with (h) heads, you typically:

1. Compute big projections $(Q = X W_Q, K = X W_K, V = X W_V)$ where $(W_Q\in\mathbb{R}^{D_{in}\times (h d)})$ etc., then reshape to $((N,h,d))$.
2. For each head (t) compute head outputs $(O^{(t)} = A^{(t)} V^{(t)})$ and get gradients $(G_{O^{(t)}})$. Apply the per-head derivations above producing $(G_{Q^{(t)}},G_{K^{(t)}},G_{V^{(t)}})$.
3. Concatenate head gradients and map back through the big $(W_Q,W_K,W_V)$ linear layers (same formulas as step 3 but with concatenated shapes).

If there is a final linear projection (W_O) applied after concatenation: $(O_\text{final} = \operatorname{concat}(O^{(1)},\dots,O^{(h)}) W_O)$, then:

* Gradient w.r.t $(W_O): (G_{W_O} = O_\text{concat}^\top,G_{O_\text{final}})$.
* Gradient back to concatenated heads: $(G_{O_\text{concat}} = G_{O_\text{final}}W_O^\top)$. Then split into per-head (G_{O^{(t)}}) and continue.

---

## Compact summary (implementation-friendly)

Given upstream (G_O):

1. $(G_A \leftarrow G_O V^\top)$
2. $(G_V \leftarrow A^\top G_O)$
3. For each row (i): compute $(c_i = \sum_j G_{A}[i,j]\cdot A[i,j])$. Then $(G_S[i,:] = A[i,:] \odot (G_A[i,:] - c_i))$. (This is the softmax Jacobian product.)
4. $(G_Q \leftarrow \dfrac{1}{\sqrt{d}}, G_S , K)$
5. $(G_K \leftarrow \dfrac{1}{\sqrt{d}}, G_S^\top , Q)$

If $(Q,K,V=XW_*)$:

6. $(G_{W_Q} \leftarrow X^\top G_Q,\quad G_{W_K} \leftarrow X^\top G_K,\quad G_{W_V} \leftarrow X^\top G_V.)$
7. $(G_X \leftarrow G_Q W_Q^\top + G_K W_K^\top + G_V W_V^\top.)$

---

## Numerical stability & implementation tips

* When computing softmax gradients, use the standard trick of subtracting the rowwise max before exponentiation to avoid overflow; the backward formula above can be implemented without recomputing exponentials by storing the softmax outputs (A) and using the (c_i) scalar trick.
* Keep careful track of row vs column orientation: usually softmax is taken **over keys** for each query row.
* Use batched matrix multiplications (BLAS) for speed; avoid explicit loops over tokens in production.
* For memory savings, many libraries fuse some ops (e.g., compute $(QK^\top)$ then softmax then multiply by (V) in one kernel).

---

## Optional: derivative of loss w.r.t. an individual attention score entry

If you want an elementwise view: Suppose $(s_{i,j})$ is a single scalar entry of (S). Its effect flows to (O) via the attention weight (a_{i,:}) row. The scalar gradient is:

[
\frac{\partial L}{\partial s_{i,j}}
= \sum_{k=1}^N \left( \frac{\partial L}{\partial a_{i,k}}\right)
\left( \delta_{k,j} a_{i,k} - a_{i,k}a_{i,j} \right)
= a_{i,j} \Big( \frac{\partial L}{\partial a_{i,j}} - \sum_{k} \frac{\partial L}{\partial a_{i,k}} a_{i,k} \Big).
]

This matches the row-vector softmax Jacobian formula shown earlier.

---


