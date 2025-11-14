# CLIP embeddings — complete end-to-end theory and math (no code)


---

# 1) High-level idea (one sentence)

CLIP (Contrastive Language–Image Pretraining) learns a **shared embedding space** for images and natural-language text by training an image encoder and a text encoder so that matching image–text pairs have **high similarity** and non-matching pairs have **low similarity**. This is achieved with a symmetric contrastive (InfoNCE) loss on large batches of (image, text) pairs.

---

# 2) Inputs and tokenization — precise math view

**Image input**

* Raw image <img width="309" height="46" alt="image" src="https://github.com/user-attachments/assets/5b960350-4595-4a3b-a0ba-5e98f04d4dfe" />

* Preprocess: resize / center-crop / normalize. Treat the encoder (CNN or ViT) as a function
 <img width="144" height="58" alt="image" src="https://github.com/user-attachments/assets/f7e5f715-2779-404a-af48-2ddea8de6383" />

  that maps the image to a fixed-dimensional feature vector (before projection). $( \phi )$ denotes image encoder parameters.

**Text input**

* Raw text string (t) (caption, prompt). Tokenize using a learned subword tokenizer (e.g., BPE/BERT-style). After tokenization get a sequence of token ids:
 <img width="414" height="39" alt="image" src="https://github.com/user-attachments/assets/363f20e6-a87d-4c75-b9af-18417e42c17e" />

* Embed tokens and process with a transformer (g_\theta) to produce a fixed vector:
 <img width="242" height="60" alt="image" src="https://github.com/user-attachments/assets/5483aef0-f181-49dc-8a00-1bb70ef4eee4" />

  where $(\theta)$ denotes text encoder parameters.

Important: both $(D_{\text{img}})$ and $(D_{\text{text}})$ are encoder output dims; CLIP then **projects** them into a common final embedding dimension (D).

---

# 3) Encoders → projection heads → normalized embeddings (math)

Let batch size be (N). We have a minibatch $({(\mathbf{I}*i, t_i)}*{i=1}^N)$.

## 3.1 Image branch

1. Extract image features:
   <img width="252" height="64" alt="image" src="https://github.com/user-attachments/assets/bc31a2c9-1a9e-48af-8606-d4a0d1b7e512" />

   (If the image encoder is a ViT, (y_i) might be the final embedding of a class token or a pooled vector; if a CNN, a global-pooled feature.)

2. Project to joint space with a linear layer:
  <img width="333" height="52" alt="image" src="https://github.com/user-attachments/assets/3d099486-abc8-4ec9-88f2-ceec7376c2d5" />

   $(W_{\text{img}}\in\mathbb{R}^{D\times D_{\text{img}}})$.

3. Normalize (L2) to unit length:
  <img width="433" height="96" alt="image" src="https://github.com/user-attachments/assets/8cb9a0a8-6ded-4206-a924-f7a86bb39769" />


## 3.2 Text branch

1. Compute text features:
  <img width="246" height="51" alt="image" src="https://github.com/user-attachments/assets/b218d65f-97fc-4c3e-accc-eb3a471c9b32" />

   (Commonly this is the transformer output pooled to a single vector — e.g., representation of end token or specially learned pooled vector.)

2. Project:
  <img width="335" height="93" alt="image" src="https://github.com/user-attachments/assets/f00a8c2b-35ac-4125-8a9c-93d8d58a8085" />

   $(W_{\text{text}}\in\mathbb{R}^{D\times D_{\text{text}}})$.

3. Normalize:
    <img width="396" height="111" alt="image" src="https://github.com/user-attachments/assets/a586988f-8e61-4bcc-a365-a00d02aed418" />


Normalization makes the model operate with cosine similarity (inner product of unit vectors equals cosine).

---

# 4) Similarity matrix and temperature scaling

For the minibatch, collect normalized vectors into matrices:

* Image matrix: $(Z^{\text{img}} = [z_1^{\text{img}}; \dots; z_N^{\text{img}}] \in \mathbb{R}^{N\times D})$.
* Text matrix:  $(Z^{\text{text}} = [z_1^{\text{text}}; \dots; z_N^{\text{text}}] \in \mathbb{R}^{N\times D})$.

Compute pairwise similarities (cosine because of normalization) as the matrix product:

<img width="657" height="112" alt="image" src="https://github.com/user-attachments/assets/7ba508e5-64fb-49f7-bfb5-bdf1f7519b5f" />


Introduce a learned or fixed temperature parameter (\tau>0) (often learned as a scalar). Scale logits by (1/\tau):
<img width="355" height="86" alt="image" src="https://github.com/user-attachments/assets/d276bbe2-bc11-47e1-887d-6147fd0e66e0" />


Intuition: smaller $(\tau)$ → sharper softmax (harder discrimination). Learning $(\log\tau)$ is common.

---

# 5) Contrastive (symmetric InfoNCE) loss — exact formula

For a minibatch of (N) matched pairs $((\mathbf{I}_i,t_i))$ assume *in-batch negatives* (every other text in the batch is a negative for image (i)). Define softmax over the *text* axis for each image:

* Probability that text (j) is the correct caption for image (i):
 <img width="326" height="70" alt="image" src="https://github.com/user-attachments/assets/dd559503-3d63-4e0b-8e58-f15a4e853f8b" />


Similarly, softmax over images for each text (symmetric direction):

<img width="307" height="82" alt="image" src="https://github.com/user-attachments/assets/0f585ae3-cb0e-45da-b8b6-3353ba9c2031" />


The CLIP loss is the average of the cross-entropy losses in both directions (image→text and text→image):

<img width="495" height="88" alt="image" src="https://github.com/user-attachments/assets/d6341ddd-3d86-4c4d-b979-fcbfa7636fc0" />

Equivalently, with matrix notation, for the image→text term:
<img width="380" height="87" alt="image" src="https://github.com/user-attachments/assets/2b1a9471-19b4-4e3f-bd09-7fee44299f30" />

And similarly for text→image.

Remarks:

* This is a form of **InfoNCE** / softmax contrastive loss. It encourages diagonal entries $(\tilde S_{ii})$ (matched pairs) to be larger than off-diagonal ones.
* Using both directions makes training symmetric and helps both encoders learn.

---

# 6) Why normalization + temperature + dot product?

* L2 normalization turns similarity into cosine similarity: $( \langle z_i,z_j\rangle = \cos(\theta)\in[-1,1])$. This bounds logits and stabilizes training.
* Temperature (\tau) controls softmax sharpness—learning $(\tau) allows the model to adapt the effective concentration of similarities.
* Using dot product (matrix multiply) is efficient on batched GPUs and naturally yields all pairwise similarities.

---

# 7) Gradients of the contrastive loss — compact derivation

We’ll derive gradients w.r.t. the image embedding $(z_i^{\text{img}})$ (unit norm) for the image→text term; text direction is analogous.

<img width="829" height="73" alt="image" src="https://github.com/user-attachments/assets/0b841b67-8a05-44ca-abe7-fef51d250367" />

The image→text loss for sample (i) is:
<img width="437" height="88" alt="image" src="https://github.com/user-attachments/assets/54b18dad-fe69-4ae6-9539-892e96e25f08" />


<img width="939" height="312" alt="image" src="https://github.com/user-attachments/assets/f4a6b029-710f-40cb-9537-ca2be27f8fc0" />


Interpretation: gradient pushes $(z_i^{\text{img}})$ toward a convex combination of text vectors weighted by probabilities, and pulls it away from its matched text vector (so overall it increases match score relative to others). The 1/τ factor scales gradients.

Because of normalization constraints (we normalized vectors before similarity), gradients must be backpropagated through the projection and normalization steps — e.g., if we used $(z = \frac{\tilde z}{|\tilde z|})$, then
<img width="261" height="109" alt="image" src="https://github.com/user-attachments/assets/bdd2202c-74c7-45df-836b-331c09526aac" />

(where $(I - zz^\top)$ is the Jacobian of unit-norm projection). Then backprop continues into $(W_{\text{img}})$ and the image encoder $(f_\phi)$.

The text→image loss yields a gradient on text embeddings:
<img width="426" height="98" alt="image" src="https://github.com/user-attachments/assets/408f0ed8-22b4-4d54-bbb2-344ea55fc644" />

where $(q_{ki})$ are the softmax probabilities in the column softmax.

Combined symmetric loss sums both gradients.

---

# 8) Batch composition and negatives

* CLIP typically uses **large minibatches** (thousands or tens of thousands across distributed workers) so that each batch provides many in-batch negatives for contrastive learning.
* On multi-GPU training, negatives are aggregated across devices (e.g., via all-gather) so each device’s batch acts as negatives for others.
* Larger negative pools improve the InfoNCE estimator and generally improve representation quality.

---

# 9) Optimization & practical training choices (math & rationale)

* Optimizer: AdamW (decoupled weight decay) or variants. Weight decay helps avoid overfitting to text/image idiosyncrasies.
* Learning rate schedule: linear warmup then cosine decay or polynomial decay. Warmup stabilizes large-batch training.
* Regularization: dropout in encoders, stochastic depth, label smoothing sometimes applied to stability.
* Temperature (\tau): often parameterized as $(\log\tau)$ and learned (so gradients update it).
* Data: massive noisy image–text datasets (web-scale) are used. Data curation and deduplication improve robustness.

---

# 10) Architecture details (mathal roles)

**Image encoder $(f_\phi)$**

* If CNN: feature map $(F\in\mathbb{R}^{C'\times H'\times W'})$, global pooling (avg or learned) to get $(y\in\mathbb{R}^{D_{\text{img}}})$.
* If ViT: split into patches, token embeddings, transformer layers; use specific pooled token or linear readout to get (y).

**Text encoder $(g_\theta)$**

* Token embedding matrix $(E\in\mathbb{R}^{V\times d_{\text{tok}}})$; positional embeddings $(P\in\mathbb{R}^{L\times d_{\text{tok}}})$.
* Transformer mapping with multi-head attention and MLPs. For a text sequence of tokens, final pooled vector (e.g., last token embedding or a learned pooling) gives $(u\in\mathbb{R}^{D_{\text{text}}})$.

Both encoders are free to have different architectures, depths and internal dimensions — the crucial point is the projection to the same final space (\mathbb{R}^D) and the contrastive objective.

---

# 11) Inference uses (math)

Given trained encoders and projection heads and temperature $(\tau)$:

## 11.1 Zero-shot image classification (the CLIP trick)

To classify image (I) into one of (C) classes using only natural language class names:

1. For each class (c) create one or many class **prompts** (t_c) (e.g., templates: `"a photo of a {}"`). Optionally average embeddings across multiple templates.

2. Compute normalized text embeddings for class (c):

<img width="245" height="84" alt="image" src="https://github.com/user-attachments/assets/2dfca670-6dde-4192-89cc-92f6e114e113" />


3. Compute image embedding:
 <img width="227" height="78" alt="image" src="https://github.com/user-attachments/assets/dc27aed0-8647-455a-a7b9-392320cf8cfa" />

4. Compute logits for classes via cosine similarity scaled by $(1/\tau)$:
  <img width="246" height="63" alt="image" src="https://github.com/user-attachments/assets/ac5c42a8-6b55-45dc-aa14-6ad5863e58fa" />

   Then softmax across (c) yields probability distribution over classes. This is how CLIP performs zero-shot classification without finetuning.

## 11.2 Retrieval

* Image → retrieve matching captions: compute (z^{\text{img}}), compute similarities with a database of $(z^{\text{text}})$, rank by dot product.
* Text → retrieve images analogously.

---

# 12) Theoretical justification (InfoNCE as mutual information estimator)

* InfoNCE can be seen as maximizing a lower bound on the mutual information (I(X;T)) between image representation and text representation (for positive pairs) given the negative sampling scheme. Thus contrastive training encourages representations that preserve information predictive of the other modality.
* Practically: the softmax contrastive objective implicitly constructs a classifier that discriminates the correct text among many negatives conditioned on the image; training that classifier forces the encoders to produce informative, aligned features.

---

# 13) Extensions & common variants (math motifs)

* **Temperature annealing or learning:** $(\tau)$ can be learned (via gradient descent) or scheduled — it controls the effective entropy of the softmax.
* **Hard negative mining**: weighting or selecting negatives with high similarity to positives can strengthen the signal (mathematically modifies the negative set used in the denominator of InfoNCE).
* **Augmented contrastive losses**: incorporate multiple views, augmentations, or extra modalities.
* **Masked language/image modeling hybrids**: combine contrastive objective with reconstruction (MAE/BEiT) for richer pretraining.
* **Multilingual text encoders**: learn cross-lingual alignments by including captions in multiple languages.

---

# 14) Practical numerical notes and stability

* Because similarities are between $([-1,1])$, the scaling by (1/\tau) matters a lot. If (\tau) is too large, gradients vanish (softmax uniform); too small → very peaky gradients and instability.
* Normalization of projected vectors is important: it decouples vector length (which could reflect confidence) from angular similarity and stabilizes training.
* When backpropagating through the normalization, remember Jacobian structure: projection onto sphere uses $( (I - z z^\top) / |\tilde z|)$.

---

# 15) Summary checklist (how the pipeline runs, step-by-step)

Training forward pass for one minibatch:

1. Tokenize texts $({t_i})$ and preprocess images $({\mathbf{I}_i})$.
2. Compute $(y_i = f_\phi(\mathbf{I}*i)), (u_i = g*\theta(t_i))$.
3. Project: $(\tilde z_i^{\text{img}} = W_{\text{img}} y_i), (\tilde z_i^{\text{text}} = W_{\text{text}} u_i)$.
4. Normalize: $(z_i^{\text{img}} = \tilde z_i^{\text{img}} / |\tilde z_i^{\text{img}}|)$, likewise for text.
5. Compute similarity matrix $(S = Z^{\text{img}}(Z^{\text{text}})^\top)$.
6. Scale by $(1/\tau)$, compute softmax rows & columns, compute symmetric InfoNCE loss $(\mathcal{L})$.
7. Backprop: compute gradients through similarity → projections → encoder parameters; update via AdamW/SGD.

Inference (zero-shot classification or retrieval) is the same minus loss and gradient: compute embeddings, compare via cosine similarity scaled by $(1/\tau)$.

---

