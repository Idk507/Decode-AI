# DALL·E embeddings 
---

# 1 — Big picture (two paradigms)

There are two closely related families of DALL·E systems:

1. **DALL·E (v1) — discrete-token autoregressive approach**

   * Convert image → discrete tokens via a learned codebook (VQ/dVAE).
   * Train a big autoregressive transformer to model $(p(\text{img_tokens} \mid \text{text}))$.
   * Sample tokens autoregressively, then decode tokens → pixels with the VQ decoder.

2. **DALL·E-2 style — embedding + diffusion approach**

   * Use a **text encoder** (often CLIP text encoder) to produce a text embedding (c).
   * Learn a **prior** $(p(z_{\text{img}} \mid c))$ mapping text embedding → image embedding. The prior can be an autoregressive transformer or a **diffusion model** that outputs CLIP-like image embeddings.
   * Use a **decoder** that maps the sampled image embedding (or a low-res latent) to pixels — commonly a diffusion model (super-resolution upsampler) or diffusion-based decoder that conditions on the embedding.

Both systems rely on well-trained encoders/decoders and a robust conditional generative model.

---

# 2 — Text encoder (tokenization → embedding)

**Input:** text string (t).

1. Tokenize: convert to token ids (w_{1:L}) using a tokenizer (BPE/subword).
2. Pass tokens through a Transformer text encoder $(g_\theta)$:
  <img width="270" height="44" alt="image" src="https://github.com/user-attachments/assets/edb2952e-73c5-450a-8710-c6970da00db8" />

   and obtain a final pooled vector $(c = g_\theta(t) \in \mathbb{R}^D)$. (In CLIP this is the CLIP text embedding.)
3. Optionally normalize (c) (e.g., L2 norm) depending on whether the upstream model expects cosine similarity.

No special math beyond standard transformer equations (attention, MLP, LayerNorm) — see ViT/Transformer derivation for those formulas. For DALL·E-2 the text embedding is treated as auxiliary conditioning variable (c).

---

# 3 — Image tokenization (VQ / dVAE) — discrete latent approach (DALL·E v1)

Original DALL·E uses a discrete image tokenizer — a learned codebook — so images become sequences of discrete symbols the transformer can model.

## 3.1 Encoder / quantizer

<img width="857" height="477" alt="image" src="https://github.com/user-attachments/assets/9bf5543e-03a7-40fa-9012-e4a36026ec89" />


## 3.2 Decoder (reconstruction)

Decoder (D_\psi) maps quantized latents back to pixels:
<img width="159" height="69" alt="image" src="https://github.com/user-attachments/assets/da05d11b-a90f-4d86-8da8-6ec602f870f7" />


## 3.3 Losses for VQ (commitment + recon)

Standard VQ-VAE training uses a combination:

<img width="1000" height="391" alt="image" src="https://github.com/user-attachments/assets/6564bb18-b2f5-4e8e-85fd-363c3d38d64d" />


After training, (x) ↔ sequence of code indices (k^*(u,v)). Flatten latents spatially into a sequence (y_{1:M}) where $(M=h\cdot w)$ (or tiled ordering).

**Important:** DALL·E used a *discrete VAE (dVAE)* variant with soft assignments and a learned prior; details vary but the math essence is codebook quantization + reconstruction + commitment.

---

# 4 — Autoregressive transformer modeling (DALL·E v1)

With image tokens (y_{1:M}) and text tokens (w_{1:L}), DALL·E trains a single large autoregressive transformer to model the joint sequence with text first:

Concatenate tokens: ([w_1,\dots,w_L,, y_1,\dots,y_M]). Model next-token probabilities autoregressively:
<img width="387" height="96" alt="image" src="https://github.com/user-attachments/assets/e3534370-9ba0-4e0d-8ba4-4100b60be643" />

where (s_t) indexes either a text token or an image token.

Conditioned image generation corresponds to sampling (y_{1:M}) from the distribution
<img width="410" height="105" alt="image" src="https://github.com/user-attachments/assets/e1844700-48d5-4c6c-88ed-0aa0a939537b" />


### Loss (cross-entropy)

Train by minimizing negative log likelihood (NLL):
<img width="467" height="100" alt="image" src="https://github.com/user-attachments/assets/f73a2a6e-8380-4140-8e7c-6ed2f70f4c01" />

This is implemented as per-token softmax cross-entropy over the combined vocabulary (text vocab + image codebook).

### Sampling (inference)

To generate images given prompt (w_{1:L}):

1. Feed (w_{1:L}) to model; then sample image token $(y_1\sim p_\theta(y_1|w_{1:L}))$.
2. Iteratively sample $(y_i\sim p_\theta(y_i|w_{1:L},y_{<i})) until (i=M)$.
3. Reshape token indices to grid and decode: $(\hat x = D_\psi({e_{k^*}}))$.

You can use temperature (T) (softmax temperature), top-k or nucleus sampling during autoregressive sampling:
<img width="232" height="68" alt="image" src="https://github.com/user-attachments/assets/e2eaacae-f133-4c7b-8574-b056abac08cc" />


**Tradeoffs:** AR transformer models long range dependencies well and supports complex joint modeling, but autoregressive sampling over (M) tokens is sequential and expensive for high-res images.

---

# 5 — DALL·E-2 style: embeddings + diffusion (modern pipeline)

DALL·E-2 moved away from direct token autoregression over VQ tokens and uses **image embeddings** (CLIP), a learned **prior** to map text→image embeddings, and a **diffusion decoder** to convert embeddings → pixels. The main components:

* CLIP (or equivalent): text encoder $(g_\theta(t))$ and image encoder $(f_\phi(x))$ producing embeddings in a common space.
* **Prior** $(p_\psi(z \mid c))$: given text embedding $(c = g_\theta(t))$, produce an image embedding (z) (the embedding that CLIP would produce for a matching image). The prior can be a diffusion model (probabilistic) or an autoregressive model.
* **Decoder** $(p_{\mathrm{dec}}(x\mid z))$ : conditional generative model (often diffusion) mapping image embedding (z) → pixels (or low-res latent → high-res image via super-resolution).

I’ll now present the math for the diffusion prior and decoder (the dominant modern approach).

---

## 5.1 CLIP embeddings (brief recap)

* Text embedding: $(c = g_\theta(t)\in\mathbb{R}^D)$.
* Image embedding: $(z_{\text{CLIP}} = f_\phi(x)\in\mathbb{R}^D)$.
* Both often L2 normalized so $(|c|=|z_{\text{CLIP}}|=1)$ and similarity is cosine.

In DALL·E-2 the *prior* models a conditional distribution over CLIP image embeddings (z_{\text{CLIP}}) given (c).

---

## 5.2 Denoising Diffusion Probabilistic Model (DDPM) — math primer

A diffusion model defines:

**Forward process (fixed):** add Gaussian noise in (T) steps to (z_0) to obtain (z_t):
<img width="742" height="241" alt="image" src="https://github.com/user-attachments/assets/8944fa59-5e98-47e9-b641-0337e79e58b6" />


**Reverse / generative model (learned):** approximate
<img width="523" height="57" alt="image" src="https://github.com/user-attachments/assets/821b413c-0dd3-4c7a-909d-9b6cf0afc50e" />


A practical parameterization predicts the noise $(\epsilon_\theta(z_t,t,c))$ such that
<img width="544" height="87" alt="image" src="https://github.com/user-attachments/assets/63a643fd-ba1d-4fa1-a170-51af8704abee" />

Training objective (denoising loss) simplifies to:
<img width="1061" height="321" alt="image" src="https://github.com/user-attachments/assets/bdee068c-2369-4ea5-97be-739249d7b890" />

---

## 5.3 Prior: modeling (p(z_{\text{CLIP}}\mid c))

Treat $(z_{\text{CLIP}})$ (the CLIP image embedding) as the target (z_0). The prior is a conditional diffusion model that learns to sample (z_0) given (c).

**Training prior:**
<img width="580" height="87" alt="image" src="https://github.com/user-attachments/assets/2a9be30b-4570-4a9f-8877-09a866920e0a" />

where <img width="210" height="33" alt="image" src="https://github.com/user-attachments/assets/8b051b1e-61e7-4287-9b27-76b9b869f210" />
, and (z_t) is constructed via the forward noising formula above.

**Sampling prior (inference):**

1. Sample $(z_0 \sim p_\psi(z_0\mid c))$ by running reverse diffusion conditioned on (c).
2. Use that sampled (z_0) as the target embedding for the decoder.

This prior yields a *distribution of plausible image embeddings* consistent with the prompt (c).

---

## 5.4 Decoder / Super-resolution: $(p_{\mathrm{dec}}(x\mid z))$

DALL·E-2 uses a diffusion-based decoder (or upsampler) to map from embedding (z) (or low-res latent) to pixels.

One approach:

1. Map (z) to a low-res image latent (y) (e.g., latent of a latent diffusion model) or treat (z) as the conditioning vector for a diffusion decoder.
2. Train a conditional diffusion model $(p_\phi(x\mid z))$ that predicts (x) (or high-res image) given (z). The standard DDPM training formula applies, replacing conditioning input (c) by (z):
<img width="410" height="59" alt="image" src="https://github.com/user-attachments/assets/9163dfeb-2091-41ad-86d9-da8351ecb817" />

3. At inference, run sampling of diffusion conditioned on sampled (z) from the prior.

A typical two-stage decoder:

* Stage A: generate a low-resolution image conditioned on (z).
* Stage B: run a super-resolution diffusion model conditioned on the low-res result and (z) to produce final high-res image.

---

# 6 — Conditioning and guidance (classifier-free guidance math)

To steer generation towards the condition (text) more strongly, DALL·E-2 uses **classifier-free guidance** — a popular technique for conditional diffusion.

Given a conditional model $(\epsilon_\theta(z_t,t,c))$ we also train an **unconditional** model $(\epsilon_\theta(z_t,t,\varnothing))$ by occasionally dropping conditions during training (e.g., with probability (p_{\text{drop}}) set (c=\varnothing)). During sampling we form a guided prediction:

<img width="567" height="74" alt="image" src="https://github.com/user-attachments/assets/9e62a50b-91ec-40cf-92ba-ef05bba51b64" />

where (w) is the guidance scale ((w>1) pushes samples toward the condition). Equivalently:
<img width="466" height="57" alt="image" src="https://github.com/user-attachments/assets/0dab4bb5-0914-4b84-9b7a-26c45b21c028" />


Plug $(\epsilon_{\text{guided}})$ into the reverse step formula to sample more strongly conditioned outputs. This trick approximates a conditional score obtained by combining unconditional and conditional score estimates, and often dramatically improves adherence to the prompt while sacrificing some diversity.

**Note:** analogous guidance exists for autoregressive models (e.g., beam search / temperature / classifier-based guidance), but classifier-free guidance is specific and powerful for diffusion.

---

# 7 — Full DALL·E-2 end-to-end training recipe (math blocks)

1. **Train CLIP (or use pre-trained)**: minimize contrastive loss to get $(f_\phi)$ and $(g_\theta)$. (See CLIP math: InfoNCE)

2. **Train decoder (diffusion) $(p_{\mathrm{dec}}(x\mid z))$**:
<img width="773" height="164" alt="image" src="https://github.com/user-attachments/assets/f00074d9-d40e-4fa8-9a6d-d1e1c3c5e4e7" />

3. **Train prior $(p_\psi(z\mid c))$ **:

 <img width="762" height="166" alt="image" src="https://github.com/user-attachments/assets/60ab13c5-3ef8-4dce-86a8-bb1da9677ee4" />


4. **(Optional) jointly fine-tune or freeze CLIP** depending on desired behavior.

**Total loss** (summed or weighted):
<img width="447" height="70" alt="image" src="https://github.com/user-attachments/assets/74e19ba4-d800-4e94-a3e6-0f56d0f94ee7" />


---

# 8 — Inference (sampling pipeline math)

To produce an image given text prompt (t):

1. Compute $(c = g_\theta(t))$ (text embedding).
2. Sample image embedding $(z \sim p_\psi(z\mid c))$ using the prior diffusion reverse process. Use classifier-free guidance if desired:

   * For each reverse step, use $(\epsilon_{\text{guided}})$ as described.
3. Use decoder $(p_{\mathrm{dec}}(x\mid z))$ (diffusion decoder) to sample (x) conditioned on (z) — again optionally guided.
4. Optionally run super-resolution steps to upscale to final resolution.

This two-stage stochastic sampling is where creative diversity arises: prior samples multiple (z) s for the same (c), and decoder maps each (z) to an image.

---

# 9 — More math: objective interpretations & evaluation

* **InfoNCE (CLIP)**: trains aligned embeddings by maximizing mutual information lower bound; gradients derived earlier.
* **DDPM loss**: denoising loss equals weighted squared error of noise prediction; under mild assumptions minimizing it matches variational lower bound on data likelihood.
* **Autoregressive NLL** (DALL·E v1): exact maximum likelihood for discrete token sequences.

**Evaluation metrics**:

* FID (Fréchet Inception Distance) — compares statistics of generated & real images in some feature space.
* CLIP similarity / CLIP-based metrics — how well images match prompts (but can be gamed).
* Human evaluation for fidelity and prompt adherence.

---

# 10 — Practical tricks & hyperparameters (mathic reasoning)

* **Temperature / sampling hyperparameters**:

  * AR models: sampling temperature (T) modifies logits $(p \propto e^{\log p / T})$.
  * Diffusion: number of steps (T), noise schedule $({\beta_t})$, and guidance scale (w) trade off fidelity vs diversity.
* **Batching for prior**: modeling CLIP embeddings (low-dim) is cheaper than pixels; prior can be trained on large batches.
* **Normalization**: CLIP embeddings often normalized; diffusion in embedding space often uses unbounded gaussian parameterization — if (z) are normalized, you may scale them before diffusion (e.g., map unit-norm to $(\mathbb{R}^D)$ via learned affine).
* **Classifier-free guidance probability**: train with condition dropped with probability $(p_{\text{drop}})$ (e.g., 0.1). This affects stability and guidance effectiveness.

---

# 11 — Advantages & limitations (math perspective)

**Advantages:**

* Discrete-token AR approach models joint distribution exactly (MLE), but is slow at sampling.
* Diffusion prior + diffusion decoder separates concerns: prior captures semantic diversity in embedding space (cheap), decoder handles pixel synthesis (costly but conditional, can be super-resolved).
* Classifier-free guidance gives a simple, mathematically justified way to trade fidelity vs diversity.

**Limitations:**

* Diffusion sampling is computationally heavy (many eval steps).
* Quality depends on CLIP alignment and dataset scale.
* Guidance amplifies mode collapse risk (over-concentrates on high-score modes).
* Autoregressive token approach requires large codebook and long sequences for high resolution.

---

# 12 — Compact checklist / equations summary

* **Text encoding:** $(c = g_\theta(t))$.
* **Image encoding (CLIP):** $(z = f_\phi(x))$.
* **VQ quantization (dVAE):** $(z_q(u,v) = e_{k^*},; k^*=\arg\min_k |z_e-e_k|)$.
* **AR modeling (DALL·E v1):** $(p(y_{1:M}\mid w_{1:L}) = \prod_i p(y_i\mid w_{1:L},y_{<i}))$. Loss: NLL.
* **DDPM forward:** $(q(z_t\mid z_0)=\mathcal N(\sqrt{\bar\alpha_t} z_0,\ (1-\bar\alpha_t)I))$.
* **DDPM training:** minimize $(\mathbb{E}|\epsilon - \epsilon_\theta(z_t,t,c)|^2)$.
* **Classifier-free guidance:** $(\epsilon_{\text{guided}} = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing))$.
* **Decoder conditional diffusion:** $(\epsilon_\phi(x_t,t,z))$ trained via denoising loss.

---

# 13 — Where to read / implement next

* For discrete-token AR approach: read VQ-VAE / dVAE papers + Transformer language modeling (attention math).
* For diffusion prior & decoder: read DDPM, improved DDPM, and classifier-free guidance papers (Ho et al., Nichol & Dhariwal, Ho et al. guidance follow-ups).
* For CLIP: InfoNCE, contrastive learning math (see earlier CLIP exposition).

---



