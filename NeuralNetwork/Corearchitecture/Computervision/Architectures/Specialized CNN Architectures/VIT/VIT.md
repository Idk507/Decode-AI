
# 1 — High level idea (one sentence)

A Vision Transformer treats an image as a sequence of flattened patches (like tokens in language), maps them to embeddings, and runs standard Transformer encoder blocks (multi-head self-attention + MLP) over that sequence; a special **classification token** summarizes the image for supervision.

---

# 2 — From image to sequence: patch embedding

**Input:** image (X \in \mathbb{R}^{H\times W \times C}) (height (H), width (W), channels (C), e.g., RGB (C=3)).

1. Choose patch size (P) (common: 16). Assume (H, W) divisible by (P).

2. Number of patches: (N = \frac{H}{P}\cdot\frac{W}{P}).

3. Partition image into non-overlapping patches. Index patches (i=1\ldots N). Each patch is a small image patch:
   [
   x_i^\text{patch} \in \mathbb{R}^{P\times P \times C}.
   ]

4. Flatten each patch to a vector:
   [
   x_i = \operatorname{vec}(x_i^\text{patch}) \in \mathbb{R}^{P^2 C}.
   ]

5. Linear projection (patch embedding): map flattened patch to (D)-dim embedding via a learnable matrix (E):
   [
   z_i^{(0)} = E, x_i + b_E \in \mathbb{R}^D,
   ]
   where (E\in\mathbb{R}^{D\times P^2 C}). This is equivalent to a convolution with kernel size (P) and stride (P) followed by flattening.

6. Add a special **classification token** (z_\text{cls}^{(0)} = \ell \in \mathbb{R}^D) (learnable). The transformer input sequence becomes:
   [
   Z^{(0)} = \big[ z_\text{cls}^{(0)}; z_1^{(0)}; z_2^{(0)}; \dots; z_N^{(0)} \big] \in \mathbb{R}^{(N+1)\times D}.
   ]

7. Positional encoding: add position information to each patch embedding so the model knows spatial arrangement. Two common choices:

   * **Learned positional embeddings** (P\in\mathbb{R}^{(N+1)\times D}): add (P_j) to token (j).
   * **Sinusoidal** (fixed) positional encodings (rare in ViT).
     Final input:
     [
     Z^{(0)} \leftarrow Z^{(0)} + P.
     ]

---

# 3 — Transformer encoder block (single layer)

A Vision Transformer stacks (L) identical encoder layers (often 12–24). Each layer (l) maps (Z^{(l-1)}\to Z^{(l)}) by:

1. **Multi-Head Self-Attention (MSA)** with residual:
   [
   \widetilde{Z}^{(l)} = Z^{(l-1)} + \operatorname{MSA}(\operatorname{LN}(Z^{(l-1)}))
   ]
   (many implementations use pre-norm: LayerNorm before each sub-layer — we’ll use pre-norm which is common in ViT variants).

2. **MLP (feed-forward network)** with residual:
   [
   Z^{(l)} = \widetilde{Z}^{(l)} + \operatorname{MLP}(\operatorname{LN}(\widetilde{Z}^{(l)}))
   ]

Where (\operatorname{LN}(\cdot)) is Layer Normalization applied token-wise.

We'll now expand MSA and MLP mathematically.

---

## 3.1 — LayerNorm

For a token vector (u\in\mathbb{R}^D):
[
\operatorname{LN}(u) = \frac{u - \mu(u)}{\sqrt{\sigma^2(u) + \epsilon}} \odot \gamma + \beta,
]
where (\mu(u)) and (\sigma^2(u)) are the mean and variance over the (D) dimensions, and (\gamma,\beta\in\mathbb{R}^D) are learnable.

---

## 3.2 — Multi-Head Self-Attention (MSA)

Given token sequence (U\in\mathbb{R}^{(N+1)\times D}) (LN applied), define (h) heads and per-head dimension (d_k = D/h).

For each head (t=1\ldots h), we have learned projection matrices:
[
W_Q^{(t)},W_K^{(t)},W_V^{(t)} \in \mathbb{R}^{D\times d_k}.
]
Compute queries/keys/values:
[
Q^{(t)} = U W_Q^{(t)} \in\mathbb{R}^{(N+1)\times d_k},\quad
K^{(t)} = U W_K^{(t)},\quad
V^{(t)} = U W_V^{(t)}.
]

Scaled dot-product attention for head (t):
[
A^{(t)} = \operatorname{softmax}!\Big(\frac{Q^{(t)} (K^{(t)})^\top}{\sqrt{d_k}} \Big) V^{(t)}.
]

* Here (\frac{QK^\top}{\sqrt{d_k}}) is an ((N+1)\times(N+1)) matrix of raw attention scores.
* Softmax is applied row-wise (softmax over keys for each query token).

Concatenate heads:
[
A = \operatorname{concat}\big(A^{(1)},\dots,A^{(h)}\big)\in\mathbb{R}^{(N+1)\times D}.
]
Final linear projection:
[
\operatorname{MSA}(U) = A W_O,\quad W_O\in\mathbb{R}^{D\times D}.
]

**Computational note:** computing (QK^\top) costs (\mathcal{O}((N+1)^2 D)) — quadratic in the sequence length (i.e., in number of patches).

---

## 3.3 — Feed-forward MLP

Per token independently, the MLP is:
[
\operatorname{MLP}(x) = W_2, \sigma(W_1 x + b_1) + b_2,
]
where (W_1\in\mathbb{R}^{D_\text{ff}\times D}), (W_2\in\mathbb{R}^{D\times D_\text{ff}}), (D_\text{ff}) is the hidden dimension (commonly (4D)), and (\sigma) is a nonlinearity (GELU used in ViT).

**GELU** (Gaussian Error Linear Unit):
[
\operatorname{GELU}(x) = x\cdot \Phi(x) \approx 0.5 x\big(1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3))\big).
]

---

# 4 — Full forward pass summary

Start with (Z^{(0)}) (patch embeddings + cls + position). For (l=1\ldots L):

1. (U = \operatorname{LN}(Z^{(l-1)})).
2. (Z' = Z^{(l-1)} + \operatorname{MSA}(U)).
3. (V = \operatorname{LN}(Z')).
4. (Z^{(l)} = Z' + \operatorname{MLP}(V)).

After (L) layers you have (Z^{(L)}\in\mathbb{R}^{(N+1)\times D}). Extract the first vector (classification token):
[
z_\text{cls} = Z^{(L)}[0] \in\mathbb{R}^D.
]

Classification head: linear layer (W_c\in\mathbb{R}^{K\times D}) (for (K) classes):
[
\hat{y} = \operatorname{softmax}(W_c z_\text{cls} + b_c).
]

Loss (for supervised classification): cross-entropy between (\hat{y}) and ground-truth label (y):
[
\mathcal{L} = -\log \hat{y}_{y}.
]

---

# 5 — Key mathematical intuitions

* **Self-attention** lets every patch attend to every other patch; the model can aggregate long-range context directly (no locality bias required).
* The **classification token** is a learnable query that pools information from all patches via attention across layers.
* **Positional embeddings** give the model spatial info lost by flattening patches.
* **Residual connections + LayerNorm** stabilize gradients and allow deep stacks.

---

# 6 — Architectural hyperparameters (and their effect)

* Patch size (P): smaller (P) → larger (N) → more tokens → more computational cost (quadratic), but finer spatial granularity. ViT-Base typically uses (P=16) or 8.
* Embedding dimension (D) (e.g., 768 in ViT-Base).
* Number of heads (h) (e.g., 12) with (d_k = D/h).
* Number of layers (L) (e.g., 12 for ViT-Base).
* MLP hidden size (D_\text{ff}) (commonly (4D)).
* Positional embedding type: learned usually works best for ViT; sinusoidal sometimes used.

---

# 7 — Training recipes and regularization (mathematical & practical)

ViT models are data-hungry; to make them work well you need specific optimizers, schedules, and regularizers.

### 7.1 Optimizer & weight decay

* Use **AdamW** (Adam with decoupled weight decay). The update for parameter (\theta) at step (t):
  [
  m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
  v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
  ]
  then bias-correct and update with weight decay (\lambda):
  [
  \theta_{t+1} = \theta_t - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_t \right).
  ]

### 7.2 Learning rate schedule

* Use **linear warmup** for first (w) steps then **cosine decay**:
  [
  \eta_t = \eta_\text{init} \cdot \begin{cases}
  t/w & t\le w\
  \frac{1}{2}\big(1 + \cos(\pi \frac{t-w}{T-w})\big) & t>w
  \end{cases}
  ]

### 7.3 Regularization used in ViT papers

* **Dropout** in MLP/attention (probability (p)).
* **Attention dropout** on softmax outputs.
* **Stochastic depth (DropPath)**: randomly drop entire residual branches during training — equivalent to scaling residual output by Bernoulli mask. Helps very deep transformers.
* **Label smoothing** in cross-entropy: replace one-hot labels by ((1-\epsilon, \epsilon/(K-1))) to regularize.
* **Data augmentations** — crucial (see next).

### 7.4 Data augmentations & mix strategies

* **RandAugment / AutoAugment**: strong image transformations.
* **MixUp**: convex combination of images and labels: (\tilde x = \lambda x_i + (1-\lambda)x_j), (\tilde y = \lambda y_i + (1-\lambda) y_j).
* **CutMix**, **RandomResizedCrop**, **RandErasing**, etc.
  Strong augmentation is essential because ViT has less built-in translation/scale invariance than CNNs.

---

# 8 — Variants & improvements (mathematical motifs)

I’ll summarize the main directions and their math intuitions (no code):

* **DeiT (Data-efficient Image Transformers):** introduce a **distillation token** and strong regularization/augmentation so ViT can be trained on ImageNet-1k (smaller dataset) with comparable results. Distillation adds an auxiliary teacher loss on an extra token.

* **Patch-based masked modeling (BEiT, MAE):** pretrain by masking a subset of patches and reconstructing them (masked image modeling). Example: randomly mask 75% patches; encoder processes visible patches; decoder reconstructs pixel or tokenized patches. Loss is reconstruction loss (e.g., L2 on pixels, or cross-entropy on discrete token ids). This significantly reduces data needs.

* **Hierarchical / windowed self-attention** (Swin Transformer): compute attention in local windows (linear complexity per window), use shifted windows to enable cross-window communication. This reduces computational complexity from (\mathcal{O}(N^2)) to (\mathcal{O}(N\cdot M)) where (M) is window size.

* **Locality bias:** add depthwise convolutional stem (ConvNet-like embedding) or relative positional bias in attention to inject locality.

---

# 9 — Complexity & scaling

* **Time / Memory complexity:** per MSA layer, cost is dominated by (QK^\top): (\mathcal{O}((N+1)^2 D)). So doubling image size or halving patch size dramatically increases cost.
* **Parameter count:** about (\mathcal{O}(L D^2)) (dominant parameters in MLP/projection matrices).
* **Compute tradeoffs:** increase (D), (L), or reduce (P) — increases accuracy but also compute/memory.

---

# 10 — Why ViT can work (and when it struggles)

**Why it works**

* Global attention from the first layer lets the model capture long-range dependencies and global structure efficiently.
* Large capacity + pretraining on large datasets (e.g., ImageNet-21k or JFT) yields strong feature learning.
* Simpler building blocks (no complicated hand-crafted CNN rules), making scaling easier.

**When it struggles**

* On **small datasets** without heavy augmentation or pretraining: tends to underperform CNNs because it lacks local inductive bias (translation equivariance) built into convs.
* **Large resolution** images blow up compute (quadratic in tokens).

---

# 11 — Interpreting ViT (attention math for visualization)

You can interpret attention matrices to see where each token attends:

For a given head (t) at layer (l), attention scores:
[
S^{(t)} = \operatorname{softmax}!\Big(\frac{Q^{(t)} (K^{(t)})^\top}{\sqrt{d_k}}\Big) \in\mathbb{R}^{(N+1)\times(N+1)}.
]
Row (i) of (S^{(t)}) is a distribution over tokens that token (i) attends to. The **cls token's** row often highlights semantically important patches; averaging or combining heads across layers can produce class-relevant saliency maps.

---

# 12 — Fine-tuning & transfer learning math

When fine-tuning a pretrained ViT on target dataset:

* Initialize weights from pretraining (encoder weights (E, W_Q, W_K, W_V, W_O,\dots)).
* Often replace/fine-tune the classification head (W_c) and possibly positional embeddings (if resolution changes).
* If input resolution differs: either interpolate learned positional embeddings to new grid (bi-linear interpolation of (P) embeddings) or use relative positional encodings.
* Fine-tune with smaller learning rate for pretrained params, larger for head.

---

# 13 — Practical checklist (mathematical/practical summary)

If you were to implement/train ViT from scratch (math checklist):

1. Patchify image → flattened patch vectors (x_i).
2. Learnable linear projection (E) to get patch embeddings (z_i^{(0)}).
3. Prepend learnable cls token and add positional embeddings (P).
4. Stack (L) Transformer encoder layers with:

   * LayerNorm
   * MSA: compute (Q,K,V), scaled dot-product, softmax, combine heads, projection (W_O).
   * Residual + MLP with GELU.
5. Take cls token output (z_\text{cls}), project to classes with linear head.
6. Train with cross-entropy, use AdamW, warmup + cosine LR, heavy augmentations (MixUp, RandAugment), dropout, label smoothing, stochastic depth.
7. If data is limited, pretrain with masked patch modeling (reconstruction) or distillation (teacher networks).

---

# 14 — Summary (concise)

* ViT replaces convolutional inductive bias by letting self-attention learn relationships across patches.
* Core math: patch linear projection, positional embeddings, multi-head scaled dot-product attention, MLP with GELU, LayerNorm, residuals.
* Strengths: global modeling, simpler scaling, strong when pretraining on lots of data or using masked modeling.
* Weaknesses: quadratic cost in tokens, needs more data or clever pretraining/regularization; many variants (DeiT, MAE, Swin) address these.

---



Which follow-up would you like?
