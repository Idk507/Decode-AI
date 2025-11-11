
---

## Notation & shapes (single head)

* Sequence length (tokens): (N).
* Input embedding dimension: (D).
* Per-head dimension: (d) (so queries/keys/values are (d)-dim).
* Query, Key, Value matrices (after linear projections):
 <img width="230" height="51" alt="image" src="https://github.com/user-attachments/assets/cd3eaca8-3a06-40df-9a2e-2319bdaa68db" />

* Scaled score matrix:
 <img width="243" height="105" alt="image" src="https://github.com/user-attachments/assets/704a89f5-1b73-4a54-be64-9d6c65673bd6" />

  (we use row (i) as query (i), columns index keys)
* Attention weights (row-wise softmax):
<img width="397" height="53" alt="image" src="https://github.com/user-attachments/assets/8f565627-3d0e-41f9-b813-a07a1beea2df" />

  where each row <img width="204" height="30" alt="image" src="https://github.com/user-attachments/assets/3dc6696b-eff7-4fa1-8f1f-3ab4c68ef421" />
sums to 1.
* Attention output:
<img width="247" height="76" alt="image" src="https://github.com/user-attachments/assets/955c7343-aefe-4d30-9aee-4b04c97a6148" />

* A scalar loss (L) depends (via later network layers) on (O). We assume we have $(\frac{\partial L}{\partial O})$, denote it $(G_O \in \mathbb{R}^{N\times d})$.

Goal: compute
<img width="742" height="142" alt="image" src="https://github.com/user-attachments/assets/0f3eacf1-df17-43ac-b2a0-e5566df8070b" />


---

## Step 0 — From output gradient to immediate upstream gradients

Because (O = A V), treat (A) and (V) as inputs to this matrix product.

1. Gradient w.r.t. (A):
 <img width="766" height="118" alt="image" src="https://github.com/user-attachments/assets/6c9537a1-6d73-46ef-af2f-90d968b1a353" />


2. Gradient w.r.t. (V):
<img width="310" height="87" alt="image" src="https://github.com/user-attachments/assets/8b29d8e6-ebed-4e2d-a002-3afd9b785557" />

   (Because (O = A V), gradient flows back to V by multiplying left by $(A^\top)$.)

So far:
<img width="407" height="93" alt="image" src="https://github.com/user-attachments/assets/b6c1d9a2-7229-4a0d-b17d-1ea4bcd927c3" />


---

## Step 1 — Backprop through softmax (row-wise)

Recall each row <img width="153" height="46" alt="image" src="https://github.com/user-attachments/assets/55cc84f4-36c9-4bc7-8229-52f195c0a124" />
where (s_i) is row (i) of (S). For a single row (i), with upstream gradient<img width="161" height="53" alt="image" src="https://github.com/user-attachments/assets/4323df14-e328-43a6-8219-7801f85cac62" />
 (row vector length (N)), the derivative w.r.t. (s_i) is:

<img width="289" height="89" alt="image" src="https://github.com/user-attachments/assets/dc4d13d5-5115-4315-ac9a-aea81b37e9b8" />

where
<img width="378" height="55" alt="image" src="https://github.com/user-attachments/assets/dfd3b0f5-299b-4f6e-905b-d1e452e283ab" />

This yields the compact and numerically stable formula (elementwise form):

For each row (i) and column (j):
<img width="511" height="98" alt="image" src="https://github.com/user-attachments/assets/e3406d44-e65c-4ad5-aa93-423a3450a979" />


Vector form for row (i):
<img width="437" height="51" alt="image" src="https://github.com/user-attachments/assets/055d5cce-6366-4081-80ae-850b4428b6cd" />

where (\odot) is elementwise multiply, and $(\mathbf{1})$ is a length-(N) vector of ones. More simply:
<img width="393" height="60" alt="image" src="https://github.com/user-attachments/assets/08f8987e-1602-456c-95f4-157fa1deb409" />


Stack across all rows: compute $(G_S\in\mathbb{R}^{N\times N})$ rowwise by the above.

So:
<img width="455" height="79" alt="image" src="https://github.com/user-attachments/assets/61428ed4-f821-4c57-849f-ef73c05d0513" />


(Implementation tip: compute rowwise scalar $(c_i = a_i^\top g^{(a)}_i)$ then $(G_S[i,:]=a_i\odot (g^{(a)}_i - c_i))$.)

---

## Step 2 — Backprop through scaled dot-product $(S = \frac{QK^\top}{\sqrt{d}})$

We have
<img width="292" height="120" alt="image" src="https://github.com/user-attachments/assets/9151e1de-eb69-4f88-b7a1-3811e37d140d" />

Differentiate:

* Gradient w.r.t. (Q):
<img width="717" height="126" alt="image" src="https://github.com/user-attachments/assets/031e6e47-fa72-4735-8d34-f57173dd7484" />


* Gradient w.r.t. (K):
<img width="444" height="91" alt="image" src="https://github.com/user-attachments/assets/78192464-4388-483b-a38d-8fecb6eb00b7" />


Proof sketch: elementwise,
<img width="291" height="95" alt="image" src="https://github.com/user-attachments/assets/a67bff65-6736-4db8-b37c-d1d1f2b97091" />

So $(\partial L/\partial Q_{i,r} = \sum_j (\partial L/\partial S_{i,j}) \cdot \frac{1}{\sqrt{d}} K_{j,r})$, which in matrix form is above.

So summarizing the chain:
<img width="448" height="247" alt="image" src="https://github.com/user-attachments/assets/7a43861b-b842-48c6-8696-9e2a3ec42906" />


---

## Step 3 — If (Q,K,V) are produced by linear layers from inputs

Commonly $(Q = XW_Q,; K = XW_K,; V = XW_V)$ where $(X\in\mathbb{R}^{N\times D_{in}})$ is the input sequence and $(W_Q,W_K,W_V\in\mathbb{R}^{D_{in}\times d})$.

Then gradients w.r.t. the learnable weight matrices are:

* $(\displaystyle \frac{\partial L}{\partial W_Q} = X^\top G_Q \in \mathbb{R}^{D_{in}\times d}.)$
* $(\displaystyle \frac{\partial L}{\partial W_K} = X^\top G_K \in \mathbb{R}^{D_{in}\times d}.)$
* $(\displaystyle \frac{\partial L}{\partial W_V} = X^\top G_V \in \mathbb{R}^{D_{in}\times d}.)$

And gradient to the input (X) accumulates contributions from all three projections:

<img width="559" height="76" alt="image" src="https://github.com/user-attachments/assets/af921182-f598-4cf4-8ac1-382abed6a620" />


(If the projections are separate per head or implemented as one big projection (XW) that is reshaped, the same matrix calculus applies with appropriate reshapes.)

---

## Step 4 — Multi-head attention notes

For multi-head attention with (h) heads, you typically:

1. Compute big projections <img width="593" height="59" alt="image" src="https://github.com/user-attachments/assets/745d9c89-e9f2-43cc-83c9-e75b0f089222" />
 etc., then reshape to $((N,h,d))$.
2. For each head (t) compute head outputs $(O^{(t)} = A^{(t)} V^{(t)})$ and get gradients $(G_{O^{(t)}})$. Apply the per-head derivations above producing $(G_{Q^{(t)}},G_{K^{(t)}},G_{V^{(t)}})$.
3. Concatenate head gradients and map back through the big $(W_Q,W_K,W_V)$ linear layers (same formulas as step 3 but with concatenated shapes).

If there is a final linear projection (W_O) applied after concatenation: <img width="385" height="45" alt="image" src="https://github.com/user-attachments/assets/63d6193c-e503-4591-a19c-19e882399587" />
, then:

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

If you want an elementwise view: Suppose $(s_{i,j})$ is a single scalar entry of (S). Its effect flows to (O) via the attention weight $(a_{i,:})$ row. The scalar gradient is:
<img width="743" height="88" alt="image" src="https://github.com/user-attachments/assets/5c3ea651-e85d-4fa6-9c5b-ee7a9ef82717" />


This matches the row-vector softmax Jacobian formula shown earlier.

---


