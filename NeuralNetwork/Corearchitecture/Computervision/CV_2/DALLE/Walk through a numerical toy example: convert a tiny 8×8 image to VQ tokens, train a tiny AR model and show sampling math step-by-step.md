
---

# 0 — Setup (what we’ll use)

* Image: 8×8 grayscale pixels with values 1..64 arranged row-major:

```
Row1:  1  2  3  4  5  6  7  8
Row2:  9 10 11 12 13 14 15 16
Row3: 17 18 19 20 21 22 23 24
Row4: 25 26 27 28 29 30 31 32
Row5: 33 34 35 36 37 38 39 40
Row6: 41 42 43 44 45 46 47 48
Row7: 49 50 51 52 53 54 55 56
Row8: 57 58 59 60 61 62 63 64
```

* We split the image into **four non-overlapping 4×4 patches** (top-left, top-right, bottom-left, bottom-right). That gives a 2×2 latent grid ⇒ **M = 4 tokens**.

* Vector quantizer (codebook) size **K = 4**. Each codebook element is a 2-D vector (so latent dimension `d = 2`).

* For simplicity our encoder produces for each 4×4 patch a 2-D vector where:

  * dimension 1 = mean of the 16 pixels in the patch,
  * dimension 2 = (mean / 10).
    (This is a toy encoder so the math is easy to follow.)

* Codebook (K=4) vectors (chosen so they are close to typical patch means):

```
e1 = [10.0, 1.0]
e2 = [15.0, 1.5]
e3 = [48.0, 4.8]
e4 = [52.0, 5.2]
```

* Order of tokens: raster order of patches → token indices for (top-left, top-right, bottom-left, bottom-right).

---

# 1 — Compute encoder outputs (z_e) for each 4×4 patch

We compute the 16-pixel mean inside each 4×4 patch.

### Top-left patch (rows 1–4, cols 1–4)

Values:

```
1  2  3  4
9 10 11 12
17 18 19 20
25 26 27 28
```

Sum row-wise:

* row1 sum = 1+2+3+4 = 10
* row2 sum = 9+10+11+12 = 42
* row3 sum = 17+18+19+20 = 74
* row4 sum = 25+26+27+28 = 106
  Total sum = 10 + 42 + 74 + 106 = **232**.
  Mean = 232 / 16 = **14.5**.

So encoder vector for patch1:

z_1 = [14.5,; 14.5/10] = [14.5,; 1.45].

---

### Top-right patch (rows 1–4, cols 5–8)

Values:

```
5  6  7  8
13 14 15 16
21 22 23 24
29 30 31 32
```

Row sums: 26, 58, 90, 122 → total = 296. Mean = 296/16 = **18.5**.
Encoder vector:

z_2 = [18.5,; 1.85].

---

### Bottom-left patch (rows 5–8, cols 1–4)

Values:

```
33 34 35 36
41 42 43 44
49 50 51 52
57 58 59 60
```

Row sums: 138,170,202,234 → total = 744. Mean = 744/16 = **46.5**.
Encoder vector:

z_3 = [46.5,; 4.65].


---

### Bottom-right patch (rows 5–8, cols 5–8)

Values:

```
37 38 39 40
45 46 47 48
53 54 55 56
61 62 63 64
```

Row sums: 154,186,218,250 → total = 808. Mean = 808/16 = **50.5**.
Encoder vector:
<img width="186" height="51" alt="image" src="https://github.com/user-attachments/assets/f3ff8b02-9774-41dd-b34d-109700e5ac24" />

So the encoder outputs (4 tokens) are:

```
z1 = [14.5, 1.45]
z2 = [18.5, 1.85]
z3 = [46.5, 4.65]
z4 = [50.5, 5.05]
```

---

# 2 — Quantize: find nearest codebook entry for each z (Euclidean distance)

We compute squared Euclidean distances (|z - e_k|^2) for k=1..4 and pick argmin.

### Patch 1: z1 = [14.5, 1.45]

Compute differences and squared distances to each e_k:
<img width="562" height="277" alt="image" src="https://github.com/user-attachments/assets/e83c0c28-a6b3-4355-a452-0a60d773155c" />


---

<img width="633" height="663" alt="image" src="https://github.com/user-attachments/assets/fd2aa217-df3b-4972-9e6f-505a47a29973" />


---

So the **quantized token sequence** (in raster order) is:

<img width="247" height="77" alt="image" src="https://github.com/user-attachments/assets/75c36fca-2e64-4cb8-bab9-6196fd8d4d6d" />


That is: top-left→2, top-right→2, bottom-left→3, bottom-right→4.

---

# 3 — Decode (reconstruction) from codebook tokens — toy decoder

For simplicity our decoder reconstructs each 4×4 patch by **filling all 16 pixels with the first coordinate** of the selected codebook vector (i.e., treat the first dimension as the patch mean). This is a simple dVAE-like decode rule for the toy.

So reconstructed patch pixel values:

* patch1 (token 2): fill with `e2[0] = 15.0` (every of the 16 pixels in patch1).
* patch2 (token 2): fill with `15.0`.
* patch3 (token 3): fill with `48.0`.
* patch4 (token 4): fill with `52.0`.

We’ll compute reconstruction error (MSE) next.

---

# 4 — Reconstruction error (MSE) computation — exact

For any patch, the per-pixel mean squared error to constant `c` equals:

<img width="536" height="91" alt="image" src="https://github.com/user-attachments/assets/f334e5d9-de8d-4097-84f7-44fd95c34354" />


We already computed each patch mean; the patch variance (of the 16 values) is the same for all 4 patches in this arithmetic-grid example — we compute it once.

### Compute variance for patch1 (we already have mean = 14.5)

We compute sum of squares for patch1:

Row squares:

* row1 squares: (1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30).
* row2: (9^2+10^2+11^2+12^2 = 81+100+121+144 = 446).
* row3: (17^2+18^2+19^2+20^2 = 289+324+361+400 = 1374).
* row4: (25^2+26^2+27^2+28^2 = 625+676+729+784 = 2814).

Total sum of squares = (30 + 446 + 1374 + 2814 = 4664).

Mean square = (4664 / 16 = 291.5).
(mean)^2 = (14.5^2 = 210.25).
Variance = mean_sq − mean^2 = (291.5 - 210.25 = \mathbf{81.25}).

(You can verify the same 81.25 appears for the other patches by analogous sums — I did that offline; the pattern holds because the image numbers form arithmetic progressions.)

Now compute per-patch MSE to reconstructed constants:

* Patch1 (reconstructed c=15.0): mean difference = 14.5 − 15.0 = −0.5. Squared = 0.25.
  MSE_patch1 = 81.25 + 0.25 = **81.5**.

* Patch2 (c=15.0, mean=18.5): mean diff = 18.5 − 15.0 = 3.5, squared = 12.25.
  MSE_patch2 = 81.25 + 12.25 = **93.5**.

* Patch3 (c=48.0, mean=46.5): diff = 46.5 − 48.0 = −1.5, squared = 2.25.
  MSE_patch3 = 81.25 + 2.25 = **83.5**.

* Patch4 (c=52.0, mean=50.5): diff = 50.5 − 52.0 = −1.5, squared = 2.25.
  MSE_patch4 = 81.25 + 2.25 = **83.5**.

Overall MSE across the full 64 pixels (average of the 4 patch MSEs, since each patch has 16 pixels):
<img width="732" height="113" alt="image" src="https://github.com/user-attachments/assets/049916cc-dad9-4738-b81a-0fc0da4358d0" />


So the toy VQ autoencoder reconstructs with MSE = **85.5** under our decoder rule.

---

# 5 — Tiny autoregressive model for the token sequence

We now model the discrete token sequence (y = [y_1,y_2,y_3,y_4] = [2,2,3,4]) with a simple **order-1 autoregressive model** (Markov of order 1). That is:

<img width="468" height="47" alt="image" src="https://github.com/user-attachments/assets/38d18432-869d-49dd-b4dc-f9e838533ed8" />

Each conditional distribution is parameterized by logits over the K=4 code indices (softmax of logits → probabilities).

We’ll give an initial toy set of logits (before training) and compute the loss and gradients for one training example, then update logits by one gradient step.

---

## 5.1 Initial model logits (toy)

* Position 1 (initial distribution over tokens): logits (a = [0.1,; 1.0,; 0.2,; -0.5]).
  (meaning scores for token indices 1..4 at position 1)

* Conditional logits for next token given previous token = 2 (we use the same parameter for both positions 2 and 3, since prev token = 2 occurs twice in our sequence): denote vector (b = [0.2,; 0.3,; -0.1,; 0.0]). This parameter is used both for $(p(y_2 \mid y_1=2))$ and $(p(y_3 \mid y_2=2))$.

* Conditional logits for next token given previous token = 3: (c = [-0.2,; 0.0,; 0.5,; 0.1]). This is used for $(p(y_4 \mid y_3=3))$.

These are toy numbers chosen so we can compute softmaxes and gradients.

---

## 5.2 Forward pass: compute probabilities and negative log-likelihood loss

We compute softmax probabilities and per-position cross-entropy losses.

### Position 1: logits (a = [0.1, 1.0, 0.2, -0.5])

Compute softmax:

<img width="814" height="425" alt="image" src="https://github.com/user-attachments/assets/80c5d15c-02cc-409f-84b3-e0d74b1b9fd1" />


Observed (y_1 = 2) (index 2). Cross-entropy loss:
<img width="458" height="37" alt="image" src="https://github.com/user-attachments/assets/1216f6f2-f855-40c7-b557-1871f29baad9" />


Gradient wrt logits (a) for cross-entropy is <img width="172" height="24" alt="image" src="https://github.com/user-attachments/assets/a91fab71-b494-42ce-b018-44c1a72fdf4b" />
. So

<img width="651" height="48" alt="image" src="https://github.com/user-attachments/assets/470d461b-54c8-4e8a-b18a-1ba03efe2b24" />

---

### Position 2: conditional logits given prev = 2: (b = [0.2,;0.3,;-0.1,;0.0])

Compute softmax:
<img width="715" height="300" alt="image" src="https://github.com/user-attachments/assets/f2420e6c-390a-496a-a70a-214bb73f1858" />


Observed (y_2 = 2). Loss:
<img width="663" height="112" alt="image" src="https://github.com/user-attachments/assets/7cd0a925-8810-466f-8896-dc1b35996af0" />

---

### Position 3: conditional logits given prev = 2 again (same `b`)

<img width="632" height="148" alt="image" src="https://github.com/user-attachments/assets/69f789f9-9500-4910-822c-28b144d15daa" />


### Position 4: conditional logits given prev = 3: (c = [-0.2, 0.0, 0.5, 0.1])

Compute softmax:

* max = 0.5 → subtract: [-0.7, -0.5, 0, -0.4].
* exp: exp(-0.7)=0.496585, exp(-0.5)=0.606531, exp(0)=1, exp(-0.4)=0.670320.
* sum = 0.496585 + 0.606531 + 1 + 0.670320 = 2.773436.
* probabilities:

  <img width="467" height="140" alt="image" src="https://github.com/user-attachments/assets/484ec9f1-a0d7-4f32-882b-60b3d1ee265e" />


Observed (y_4 = 4). Loss:
<img width="353" height="59" alt="image" src="https://github.com/user-attachments/assets/d0d65823-66db-4a65-a331-6a859b8422ee" />

Gradient (g_4 = p_4 - \text{one_hot}(4) = [0.1791, 0.2187, 0.3606, -0.7584]).

---

### Total loss (sum of position losses)

[
L = L_1 + L_2 + L_3 + L_4 \approx 0.7320 + 1.198 + 1.598 + 1.420 = \mathbf{4.948}.
]

(If training over many examples we'd average; here just one example so sum is fine.)

---

# 6 — Backprop (gradients) and a single parameter update (learning rate 0.1)

We now update the logits parameters by gradient descent. For simplicity we treat `a`, `b`, `c` directly as parameters (i.e., no upstream network). For `b`, remember it’s used twice (positions 2 and 3), so its total gradient is (g_2 + g_3).

Compute parameter gradients:
<img width="749" height="239" alt="image" src="https://github.com/user-attachments/assets/002ee6e0-3f84-4a25-b80b-11487c2af602" />


Use learning rate (\eta = 0.1). Gradient descent update: param ← param − η * grad.

### Update `a`:

* 0.1 * grad_a = [0.0195693, -0.0518939, 0.0216150, 0.0107096].
* New `a` = old `a` − 0.1*grad:

```
a_old = [0.1, 1.0, 0.2, -0.5]
a_new = [
  0.1  - 0.0195693 = 0.0804307,
  1.0  - (-0.0518939) = 1.0518939,
  0.2  - 0.0216150 = 0.1783850,
 -0.5  - 0.0107096 = -0.5107096
]
```

So
<img width="495" height="57" alt="image" src="https://github.com/user-attachments/assets/84c8590b-80cd-4b8c-a8ce-1ebdd563d802" />


---

### Update `b` (accumulated grad)

* 0.1 * grad_b = [0.05456, -0.03972, -0.05956, 0.04472].
* b_old = [0.2, 0.3, -0.1, 0.0].
* b_new = b_old − 0.1*grad_b:

```
b_new = [
  0.2  - 0.05456 = 0.14544,
  0.3  - (-0.03972) = 0.33972,
 -0.1  - (-0.05956) = -0.04044,
  0.0  - 0.04472 = -0.04472
]
```

So
<img width="490" height="70" alt="image" src="https://github.com/user-attachments/assets/3de61de1-a171-4ddd-b821-41b5b3987032" />


---

### Update `c`

* 0.1 * grad_c = [0.01791, 0.02187, 0.03606, -0.07584].
* c_old = [-0.2, 0.0, 0.5, 0.1].
* c_new = c_old − 0.1*grad_c:

```
c_new = [
 -0.2 - 0.01791 = -0.21791,
  0.0 - 0.02187 = -0.02187,
  0.5 - 0.03606 =  0.46394,
  0.1 - (-0.07584)= 0.17584
]
```

So
<img width="495" height="70" alt="image" src="https://github.com/user-attachments/assets/2c131b72-1c90-46ba-8f8c-06970dc18972" />


---

# 7 — Sampling from the updated AR model (step-by-step)

Now we’ll **sample** a sequence from the updated model parameters $(a_{\text{new}}, b_{\text{new}}, c_{\text{new}})$. We’ll compute the new softmax probabilities (exact arithmetic shown) and then take the most-likely token (argmax) to illustrate deterministic sampling — you can alternatively sample stochastically by sampling from the categorical distribution.

### Position 1 sampling (from `a_new`)

a_new ≈ `[0.0804307, 1.0518939, 0.1783850, -0.5107096]`.

Softmax:

* max = 1.0518939 → subtract: `[-0.9714632, 0, -0.8735089, -1.5626035]`.
* exp:

  * exp(-0.9714632) ≈ 0.378646
  * exp(0) = 1
  * exp(-0.8735089) ≈ 0.4179
  * exp(-1.5626035) ≈ 0.2096
* sum ≈ 0.378646 + 1 + 0.4179 + 0.2096 = **2.006146**.
* probabilities:

  * p1_new[1] = 0.378646 / 2.006146 ≈ **0.1887**
  * p1_new[2] = 1 / 2.006146 ≈ **0.4985**
  * p1_new[3] = 0.4179 / 2.006146 ≈ **0.2083**
  * p1_new[4] = 0.2096 / 2.006146 ≈ **0.1045**

Argmax is index **2** (prob ≈ 0.4985). If we take argmax sampling, $(y_1^{\text{sample}} = 2)$. If we sampled stochastically, there’s ~49.9% chance to pick 2, ~20.8% chance to pick 3, etc.

---

### Position 2 sampling (given sampled y1=2) — use `b_new`

b_new ≈ `[0.14544, 0.33972, -0.04044, -0.04472]`.

Softmax (subtract max = 0.33972):

* subtract: `[-0.19428, 0, -0.38016, -0.38444]`.
* exp: exp(-0.19428)=0.8236; exp(0)=1; exp(-0.38016)=0.6838; exp(-0.38444)=0.6808.
* sum ≈ 0.8236 + 1 + 0.6838 + 0.6808 = **3.1882**.
* probabilities:

  * p[1] ≈ 0.8236 / 3.1882 ≈ **0.2584**
  * p[2] ≈ 1 / 3.1882 ≈ **0.3138**
  * p[3] ≈ 0.6838 / 3.1882 ≈ **0.2144**
  * p[4] ≈ 0.6808 / 3.1882 ≈ **0.2134**

Argmax is index **2** (prob ≈ 0.3138). So $(y_2^{\text{sample}} = 2)$ if we choose argmax; stochastically there is ~31% chance.

---

### Position 3 sampling (given y2=2) — same `b_new`

Same probabilities as above. Argmax → index **2**, but stochastic sampling might pick index3 with probability ≈ 0.2144. For variety, suppose we sample stochastically and we happen to draw **3** (this reproduces the original sequence). If we do deterministic argmax, we’d get 2 again; both are valid sampling modes.

To illustrate both possibilities, here are two sampling routes:

* Deterministic (argmax every step): would give [2,2,2,...] — different from training sequence.
* Stochastic draw (one possible sample): [2 (pos1), 2 (pos2), 3 (pos3), ...] — we’ll continue with this route since it can match the training sequence.

Assume we sampled (y_3^{\text{sample}} = 3) (drawn with probability ≈ 0.2144).

---

### Position 4 sampling (given y3=3) — use `c_new`

c_new ≈ `[-0.21791, -0.02187, 0.46394, 0.17584]`.

Softmax:

* max = 0.46394 → subtract: `[-0.68185, -0.48581, 0, -0.28810]`.
* exp: exp(-0.68185)=0.5057; exp(-0.48581)=0.6151; exp(0)=1; exp(-0.28810)=0.7497.
* sum ≈ 0.5057 + 0.6151 + 1 + 0.7497 = **2.8705**.
* probabilities:

  * p[1] ≈ 0.5057 / 2.8705 ≈ **0.1761**
  * p[2] ≈ 0.6151 / 2.8705 ≈ **0.2143**
  * p[3] ≈ 1 / 2.8705 ≈ **0.3484**
  * p[4] ≈ 0.7497 / 2.8705 ≈ **0.2612**

Argmax is index **3** (prob ≈ 0.3484). If we sampled stochastically, there is about 26.1% chance of index 4. In our training sequence the true y4 = 4 (token 4). A stochastic draw might produce 4 with probability ≈ 0.2612.

So one possible sampled sequence (stochastically) is **[2, 2, 3, 4]** — exactly the training sequence — but many other sequences are possible.

---

# 8 — Decode the sampled sequence back to image (reconstruction) and MSE

If we sampled (y = [2,2,3,4]) (the original training sequence), decoding with the same decoder yields the same reconstructed image as earlier and the **same MSE = 85.5**.

If another sequence was sampled (e.g., [2,2,2,3] or [2,2,2,2]), the reconstructed image would differ (some patches replaced by different codebook means), and you could recompute MSE using the same variance + squared bias formula.

---

# 9 — Summary / takeaways from the toy example

* We converted an **8×8** image into **4 VQ tokens** by (1) encoding each 4×4 patch into a 2-D latent and (2) quantizing to the nearest codebook vector. The exact numeric nearest-neighbor computations were shown (Euclidean squared distances).
* The quantized sequence was (y=[2,2,3,4]).
* We showed a simple decoding rule (fill patch by codebook’s first coordinate) and computed the exact reconstruction MSE = **85.5**.
* We defined a tiny AR model (first-step logits `a`, conditional logits `b` and `c`), computed softmax probabilities, computed the sum cross-entropy loss for the observed sequence, computed the per-logit gradients (exact numbers), performed a single gradient-descent update with learning rate 0.1, and displayed the new logits.
* We then sampled from the updated model, calculating the updated softmax probabilities and showing how you might (deterministically or stochastically) produce sequences — one of which is the original training sequence.
* Everything was computed numerically and shown step-by-step, so you can reproduce these calculations exactly.

---

