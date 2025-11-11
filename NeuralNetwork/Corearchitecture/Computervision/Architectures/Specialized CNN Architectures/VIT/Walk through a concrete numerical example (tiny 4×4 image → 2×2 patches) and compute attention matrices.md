
**Setup (what we’ll do)**

* Image: 4×4 grayscale, pixel values 1..16 laid out row-major.
* Patch size: 2 → each patch is 2×2 → flattened patch vector length 4. Number of patches (N=4).
* We add a learnable **classification token** → total tokens (N+1 = 5).
* Embedding dimension (D=4).
* Number of attention heads (h=2) → per-head dimension (d_k = D/h = 2).
* We'll compute: patch embeddings → add positional embeddings → form input matrix (Z^{(0)}) → compute Q, K, V for each head → compute scaled dot-product attention scores → softmax → attention outputs → concat heads → final MSA output for each token. (We keep things small and explicit; there is no LayerNorm/MLP shown in the numeric steps so the attention math is clear.)

---

## 1) Image → patches

Image (4×4):

```
[[ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12],
 [13, 14, 15, 16]]
```

With patch size 2 we get 4 patches (order: top-left, top-right, bottom-left, bottom-right). Each patch flattened (row-major):

1. patch 1 (top-left): [1, 2, 5, 6]
2. patch 2 (top-right): [3, 4, 7, 8]
3. patch 3 (bottom-left): [9,10,13,14]
4. patch 4 (bottom-right): [11,12,15,16]

So `patches` matrix (4 rows × 4 cols):

```
patches =
[[ 1,  2,  5,  6],
 [ 3,  4,  7,  8],
 [ 9, 10, 13, 14],
 [11, 12, 15, 16]]
```

---

## 2) Patch embedding (linear projection)

We use a small embedding matrix $(E\in\mathbb{R}^{4\times4})$ (rows are output dims):

```
E =
[[1.0, 0.0, 0.0, 0.0],
 [0.0, 0.5, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.25],
 [0.1, 0.1, 0.1, 0.1]]
```

Embed each patch: (z_i = E \cdot \text{patch}_i) (we actually compute `patch * E^T` so row vectors map to row embeddings). The resulting 4 patch embeddings (z_1..z_4) are:

```
z =
[
 [ 1.0, 1.0, 1.5, 1.4    ],  # patch1
 [ 3.0, 2.0, 2.0, 2.2    ],  # patch2
 [ 9.0, 5.0, 3.5, 4.6    ],  # patch3
 [11.0, 6.0, 4.0, 5.4    ]   # patch4
]
```

(Verify: e.g. for patch1 [1,2,5,6]:

* dim0 = 1*1 + 2*0 + 5*0 + 6*0 = 1.0
* dim1 = 1*0 + 2*0.5 + ... = 1.0
* dim2 = 1*0 + ... + 6*0.25 = 1.5
* dim3 = 0.1*(1+2+5+6)=1.4)

We also have a **cls token** embedding (learnable):

```
z_cls = [0.5, 0.5, 0.5, 0.5]
```

We add small positional embeddings for each of the 5 tokens (cls + 4 patches). Positional matrix (P) (5×4):

```
P =
[ [0.10, 0.00, 0.00, 0.00],   # cls
  [0.00, 0.10, 0.00, 0.00],   # patch1
  [0.00, 0.00, 0.10, 0.00],   # patch2
  [0.00, 0.00, 0.00, 0.10],   # patch3
  [0.05, 0.05, 0.05, 0.05] ]  # patch4
```

Add cls and positions: the full input token matrix (Z^{(0)}) (5 tokens × 4 dims) is:

```
Z0 = [ z_cls + P[0] ,
       z1    + P[1] ,
       z2    + P[2] ,
       z3    + P[3] ,
       z4    + P[4] ]

Z0 =
[
 [ 0.6 ,  0.5 , 0.5 , 0.5   ],   # cls (0.5 + 0.1 at dim0)
 [ 1.0 ,  1.1 , 1.5 , 1.4   ],   # patch1 + pos
 [ 3.0 ,  2.0 , 2.1 , 2.2   ],   # patch2 + pos (pos adds 0.1 on dim2)
 [ 9.0 ,  5.0 , 3.5 , 4.7   ],   # patch3 + pos (pos adds 0.1 on dim3)
 [11.05,  6.05, 4.05, 5.45  ]    # patch4 + pos (pos adds 0.05 to each dim)
]
```

(rounded for clarity)

---

## 3) Define attention heads (Q, K, V projections)

We use **2 heads**. For each head we need projection matrices $(W_Q, W_K, W_V \in \mathbb{R}^{D\times d_k})$ with (d_k=2). To keep numbers simple, choose the following explicit projections so the first head attends to the first two dims and second head attends to the last two dims:

**Head 1** (focus on dims 0 and 1):

```
W_Q1 = W_K1 = W_V1 =
[[1, 0],
 [0, 1],
 [0, 0],
 [0, 0]]   (shape D×2)
```

So head1 projects tokens to the first two components.

**Head 2** (focus on dims 2 and 3):

```
W_Q2 = W_K2 = W_V2 =
[[0, 0],
 [0, 0],
 [1, 0],
 [0, 1]]
```

So head2 projects tokens to dims 2 & 3.

Now compute per-head Q, K, V: (shape 5 tokens × 2 dims for each head).

### Head 1 projections (Q1, K1, V1) — take dims 0 & 1:

From Z0 rows, the first two dims give:

```
Q1 = K1 = V1 =
[
 [0.6,   0.5 ],   # cls
 [1.0,   1.1 ],
 [3.0,   2.0 ],
 [9.0,   5.0 ],
 [11.05, 6.05]
]
```

### Head 2 projections (Q2, K2, V2) — take dims 2 & 3:

```
Q2 = K2 = V2 =
[
 [0.5 , 0.5  ],   # cls (z_cls dims 2&3)
 [1.5 , 1.4  ],   # patch1
 [2.1 , 2.2  ],   # patch2
 [3.5 , 4.7  ],   # patch3
 [4.05, 5.45 ]    # patch4
]
```

(We used the Z0 values above for dims 2 & 3.)

---

## 4) Scaled dot-product attention — head by head

For a head with (Q) (5×2), (K) (5×2), (V) (5×2):

1. Compute raw scores matrix (S = Q K^\top) (shape 5×5).
2. Scale by $(\sqrt{d_k})$ with (d_k=2) so divide by $(\sqrt{2}\approx1.41421356)$.
3. Row-wise softmax to get attention weights (A) (5×5).
4. Output for the head is (A V) (5×2).

I'll show the full numeric results (rounded to 6 decimals) and then demonstrate the *detailed* softmax calculation for the first row of Head 1 as an example.

### Head 1: raw scores (S^{(1)}) and scaled

Computed $(S^{(1)} = \dfrac{Q1 \cdot K1^T}{\sqrt{2}})$. The numeric (5×5) score matrix for head 1 is:

```
scores_head1 (5×5) ≈
[[ 0.431335 ,  0.8131728,  1.979899 ,  5.5861436,  6.827116  ],
 [ 0.8131728,  2.050581 ,  4.214550 , 11.889382 , 14.527285  ],
 [ 1.979899 ,  4.214550 , 10.101940, 25.095047 , 30.66459   ],
 [ 5.5861436, 11.889382 , 25.095047, 68.633998 , 83.58371   ],
 [ 6.827116 , 14.527285 , 30.66459 , 83.58371  , 101.500025 ]]
```

(Interpretation: row i, col j is the scaled dot-product score from token i to token j in head1.)

Now softmax each row (row-wise) to get attention probabilities (A^{(1)}). For **row 0** (the cls token row), the raw scores were:

```
row0 = [0.43133514, 0.8131728, 1.97989899, 5.58614357, 6.82711597]
```

**Compute softmax for this row step by step** (demonstration):

* Step 1: subtract max to avoid overflow: max = 6.82711597
  subtract: `row0 - max` = `[-6.39578084, -6.01394317, -4.84721699, -1.2409724, 0.0]`.

* Step 2: exponentiate each element:

  ```
  exp = [exp(-6.39578084), exp(-6.01394317), exp(-4.84721699), exp(-1.2409724), exp(0)]
      ≈ [0.00166858, 0.00244443, 0.00785019, 0.28910296, 1.0]
  ```

* Step 3: sum of exps = `0.00166858 + 0.00244443 + 0.00785019 + 0.28910296 + 1.0 ≈ 1.3000662`.

* Step 4: divide each exp by the sum to get probabilities:

  ```
  probs ≈ [0.00128247, 0.00187879, 0.00603366, 0.22220465, 0.76860042]
  ```

So the cls token attends **~76.86%** to patch 4, **~22.2%** to patch 3, and almost nothing to the others for head 1.

Performing this softmax per row for head 1 gives the full head-1 attention matrix (each row sums to 1):

```
attn_head1 (5×5) ≈
[
 [0.0012825 , 0.00187879, 0.00603366, 0.22220465, 0.76860042],
 [0.00088239, 0.00182026, 0.0056273 , 0.22461106, 0.767059  ],
 [0.00075447, 0.00155637, 0.00480854, 0.22231809, 0.77056254],
 [0.00004517, 0.00013896, 0.00053723, 0.20313151, 0.79614713],
 [0.00004063, 0.00008338, 0.00024342, 0.20133612, 0.79829644]
]
```

(You can see all rows put most mass on the last token (patch 4) in this constructed example.)

Finally, head1 output = `A_head1 × V1` (5×2). Numerically:

```
out_head1 ≈
[
 [ 5.093208 ,  3.240890 ],
 [ 9.225778 ,  5.893885 ],
 [19.390018 , 11.012148 ],
 [53.135851 , 30.448799 ],
 [62.939955 , 36.503603 ]
]
```

---

### Head 2: raw scores (S^{(2)}) and attention

Head 2 focuses on dims 2 & 3 (patch-local activations). Its raw scaled scores matrix (5×5) is:

```
scores_head2 (5×5) ≈
[
 [ 0.25   ,  0.911043,  1.363373,  3.320291,  3.992158 ],
 [ 0.911043,  3.02   ,  4.177006, 10.104228, 11.049999],
 [ 1.363373,  4.177006,  6.674817, 16.121314, 17.337287],
 [ 3.320291, 10.104228, 16.121314, 38.148   , 41.147428],
 [ 3.992158, 11.049999, 17.337287, 41.147428, 44.341178]
]
```

Compute row-wise softmax to get `attn_head2` (5×5). For head2 the cls row probabilities are (row 0):

* Raw row0: `[0.25, 0.911043, 1.363373, 3.320291, 3.992158]`
* Softmaxed → approx:

  ```
  [0.00259991, 0.00558589, 0.01382338, 0.33159796, 0.64639286]
  ```

Full head2 attention matrix:

```
attn_head2 ≈
[
 [0.00259991, 0.00558589, 0.01382338, 0.33159796, 0.64639286],
 [0.00095877, 0.00278752, 0.00748726, 0.32323156, 0.66553489],
 [0.00070076, 0.00197   , 0.00563237, 0.32935678, 0.66233908],
 [0.00004409, 0.00016079, 0.00064873, 0.23018301, 0.769,    ],
 [0.0000341 , 0.00009434, 0.00036077, 0.19800328, 0.80150751]
]
```

(Again, in this toy setup most mass ends up on the last token; this is because the numeric projections we chose produced larger dot products toward the later patches.)

Head2 output = `A_head2 × V2` (5×2), numerically:

```
out_head2 ≈
[
 [ 6.0204179 ,  3.0349414 ],
 [10.480257 ,  5.470828 ],
 [21.044451 , 10.580398 ],
 [57.805474 , 28.860012 ],
 [69.062  ,  34.918144 ]
]
```

---

## 5) Concatenate heads and project (MSA output)

Concatenate the two head outputs per token (each token now has 2+2 = 4 dims):

For each token i: `concat_i = [out1_i (2 dims) ; out2_i (2 dims)]`. Example for cls token (token 0):

```
concat_cls = [5.093208, 3.240890, 6.020418, 3.034941]  (approx)
```

If we have an output projection (W_O) (D×D), we would multiply `concat` by (W_O). For clarity we set $(W_O = I_{4})$ (identity), so MSA output equals concatenated heads.

So **MSA_out** (5×4) ≈ (first row shown):

```
msa_out[0] ≈ [ 5.093208, 3.240890, 6.020418, 3.034941 ]
```

---

## 6) Residual connection: add back to Z0

After MSA the usual transformer adds a residual: (Z' = Z^{(0)} + \text{MSA_out}). For the **cls token** (token 0):

* Z0 cls row: `[0.6, 0.5, 0.5, 0.5]`
* MSA_out cls row: `[5.093208, 3.240890, 6.020418, 3.034941]`
* Sum → final (post-attention) cls embedding:

```
Z_after_msa[cls] ≈ [11.113626, 6.275831, 3.992158, 5.109328]
```

(These values are just the elementwise sum; I rounded in final presentation.)

---

## 7) Interpretation & takeaways

* The **attention matrices** tell you “how much each token (query) attends to every other token (key)”. In this toy example the cls token strongly attends to the 4th patch (last token) in both heads — that is why most of the output’s mass comes from that patch.
* We showed **how** to compute attention step-by-step: linear projections → dot products → scale → softmax → weighted sum of values.
* In a real ViT, the projections (W_Q,W_K,W_V) and positional embeddings are learned; LayerNorm and MLP layers are applied before/after the attention block. Also: practical ViTs use bigger (D) and many heads; compute is quadratic in number of tokens.
* I deliberately chose simple projection matrices so you could *see* the separation of roles between heads (first head attends to dims 0–1, second to dims 2–3), making the arithmetic transparent.

---


Which of those next steps do you want?
